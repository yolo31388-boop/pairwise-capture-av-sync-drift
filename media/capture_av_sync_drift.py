"""采集音画同步漂移修复核心模块。

屏幕采集（视频）与麦克风采集（音频）各自使用设备本地时钟打时间戳，存在
时基不统一、时钟漂移、丢帧空洞、乱序、暂停恢复基准错位、热插拔时间戳归零
以及输出前缺少校验等问题。本模块在纯软件层面把两路流校正到统一主时基上：

1. ``unify_timebase`` / ``drift_estimate`` —— 统一时基并估算漂移；
2. ``gap_fill`` —— 按时间戳对空洞做线性插值补齐；
3. ``pts_sort`` —— 编码前按 PTS 重排（稳定排序，容忍乱序与重复）；
4. ``reset_base`` —— 暂停/恢复时重置时钟基准；
5. ``hotplug_align`` —— 设备热插拔后对新流做时间戳对齐（新流本地时间从零开始）；
6. ``sync_check`` / ``validation`` / ``mux_ready`` —— 输出前校验音画偏移。

所有时间单位均为秒（浮点），全部使用标准库、无第三方依赖，结果确定可复现。

数据结构约定
------------
包（packet）支持两种写法，内部统一规整为 dict：

* dict：``{"kind": "video"|"audio", "pts": <主时基秒>, "pts_local": <本地时钟秒>,
  "source": <流标识>, "frame_no"/"sample_no": <序号>, ...}``；
* 二元组：``(pts, kind)`` 或 ``(pts, kind, source)``。

锚点（anchor，同一物理瞬间在本地时钟与主时钟上的读数）：

* dict：``{"kind"/"source": ..., "local": <本地秒>, "master": <主秒>}``；
* 二元组：``(local, master)``。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "unify_timebase",
    "drift_estimate",
    "gap_fill",
    "pts_sort",
    "reset_base",
    "hotplug_align",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "video_capture",
    "metrics",
    "audio_capture",
    "mux_ready",
    "sync_check",
]

DEFAULT_VIDEO_FPS = 30.0
DEFAULT_AUDIO_HZ = 48000.0
# 输出前允许的音画偏移上限（秒），约一帧（30fps）以内。
DEFAULT_AV_TOLERANCE = 1.0 / 30.0
# 编码前重排缓冲窗口：允许最多这么多包的乱序到达。
DEFAULT_REORDER_WINDOW = 8
# 漂移/偏移显著判定的最小阈值，避免把浮点噪声当成问题上报。
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _is_seq(value: Any) -> bool:
    return isinstance(value, (list, tuple, deque))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, deque)):
        return list(value)
    return [value]


def _kind_of(packet: Dict[str, Any], default: str = "video") -> str:
    return str(packet.get("kind") or packet.get("stream") or default)


def _source_of(packet: Dict[str, Any], default: Optional[str] = None) -> Any:
    return packet.get("source", packet.get("device", default))


def _pts_of(packet: Dict[str, Any]) -> float:
    for key in ("pts", "pts_master", "master", "t", "timestamp"):
        if key in packet and packet[key] is not None:
            return float(packet[key])
    raise ValueError("packet 缺少 pts 时间戳字段")


def _local_of(packet: Dict[str, Any]) -> Optional[float]:
    for key in ("pts_local", "local", "device_ts"):
        if key in packet and packet[key] is not None:
            return float(packet[key])
    return None


def _normalize_packet(raw: Any, default_kind: str = "video") -> Dict[str, Any]:
    """把 dict / 元组 / 标量规整为统一包结构（不修改入参）。"""
    if isinstance(raw, dict):
        packet = dict(raw)
    elif _is_seq(raw):
        if len(raw) == 1:
            packet = {"pts": float(raw[0])}
        elif len(raw) == 2:
            packet = {"pts": float(raw[0]), "kind": str(raw[1])}
        else:
            packet = {
                "pts": float(raw[0]),
                "kind": str(raw[1]),
                "source": raw[2],
            }
    else:
        packet = {"pts": float(raw)}
    packet.setdefault("kind", default_kind)
    packet.setdefault("synthetic", False)
    return packet


def _normalize_anchor(raw: Any, kind: Optional[str] = None,
                      source: Any = None) -> Dict[str, Any]:
    if isinstance(raw, dict):
        anchor = dict(raw)
        local = anchor.get("local", anchor.get("t_local", anchor.get("device")))
        master = anchor.get("master", anchor.get("t_master", anchor.get("host")))
        anchor["local"] = float(local)
        anchor["master"] = float(master)
        anchor.setdefault("kind", kind or "video")
        if source is not None:
            anchor.setdefault("source", source)
    elif _is_seq(raw) and len(raw) >= 2:
        anchor = {
            "local": float(raw[0]),
            "master": float(raw[1]),
            "kind": kind or "video",
        }
        if source is not None:
            anchor["source"] = source
    else:
        raise ValueError(f"无法解析的锚点: {raw!r}")
    return anchor


def _same_stream(anchor: Dict[str, Any], packet: Dict[str, Any]) -> bool:
    a_src = _source_of(anchor)
    p_src = _source_of(packet)
    if a_src is not None and p_src is not None and a_src != p_src:
        return False
    return _kind_of(anchor) == _kind_of(packet)


def _linear_fit(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """对 (x, y) 点集做最小二乘线性拟合，返回 (斜率, 截距)。

    点数不足或 x 完全相同时退化为常数/恒等映射，保证不会抛异常。
    """
    n = len(points)
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return 1.0, float(points[0][1] - points[0][0])
    sx = sy = sxx = sxy = 0.0
    for x, y in points:
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
    denom = n * sxx - sx * sx
    if abs(denom) <= _EPS:
        return 1.0, (sy - sx) / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _nominal_interval(packets: Sequence[Dict[str, Any]],
                      nominal_hz: Optional[float]) -> Optional[float]:
    if nominal_hz and nominal_hz > 0:
        return 1.0 / float(nominal_hz)
    if len(packets) < 2:
        return None
    diffs = sorted(
        packets[i + 1]["pts"] - packets[i]["pts"]
        for i in range(len(packets) - 1)
    )
    diffs = [d for d in diffs if d > _EPS]
    if not diffs:
        return None
    # 中位数对个别异常间隔不敏感；先剔除明显大于中位数的空洞间隔，
    # 再用剩余的常规间隔簇求中位数，避免把大空洞本身当成标称间隔。
    mid = len(diffs) // 2
    median = diffs[mid] if len(diffs) % 2 else (diffs[mid - 1] + diffs[mid]) / 2.0
    regular = [d for d in diffs if d <= median * 1.5]
    if not regular:
        return median
    rmid = len(regular) // 2
    return (
        regular[rmid]
        if len(regular) % 2
        else (regular[rmid - 1] + regular[rmid]) / 2.0
    )


# --------------------------------------------------------------------------- #
# 1. 统一时基 + 漂移估算
# --------------------------------------------------------------------------- #
def unify_timebase(packets: Any, anchors: Any = None,
                   state: Optional[Dict[str, Any]] = None) -> Any:
    """把各设备本地时钟时间戳换算到统一主时基。

    对每一路流，用其锚点 ``(local, master)`` 拟合

        master = rate * local + offset

    其中 ``rate`` 即本地时钟相对主时钟的走时速率（漂移）。包上同时保留
    原始 ``pts_local`` 与换算后的 ``pts``，并记录 ``rate`` / ``offset``。

    单标量入参（无锚点上下文）原样返回，保证纯函数入口可独立调用。
    """
    if not _is_seq(packets):
        return packets

    raw_packets = _as_list(packets)
    normalized = [_normalize_packet(p) for p in raw_packets]
    anchor_list = (
        [_normalize_anchor(a) for a in _as_list(anchors)] if anchors is not None else []
    )
    if state is not None:
        for a in state.get("anchors", []):
            anchor_list.append(_normalize_anchor(a))

    # 按 (kind, source) 分组拟合，多路流互不干扰。
    groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for anchor in anchor_list:
        key = (_kind_of(anchor), _source_of(anchor))
        groups.setdefault(key, []).append(anchor)

    models: Dict[Tuple[str, Any], Tuple[float, float]] = {}
    for key, group in groups.items():
        group.sort(key=lambda a: a["local"])
        models[key] = _linear_fit([(a["local"], a["master"]) for a in group])

    def _model_for(packet: Dict[str, Any]) -> Tuple[float, float]:
        key = (_kind_of(packet), _source_of(packet))
        if key in models:
            return models[key]
        # 退而求其次：同 kind 任意来源的模型（热插拔前的旧流兜底）。
        for (k, _src), model in models.items():
            if k == _kind_of(packet):
                return model
        return 1.0, 0.0

    def _has_model_for(packet: Dict[str, Any]) -> bool:
        key = (_kind_of(packet), _source_of(packet))
        if key in models:
            return True
        return any(k == _kind_of(packet) for (k, _src) in models)

    out: List[Dict[str, Any]] = []
    for packet in normalized:
        local = _local_of(packet)
        rate, offset = _model_for(packet)
        if local is None and _has_model_for(packet):
            # 只有 PTS、没有独立本地时间戳的包（如元组形态），在存在该流
            # 锚点时把 PTS 视作设备本地读数，经锚点模型换算到主时基。
            local = _pts_of(packet)
        if local is not None:
            master = rate * local + offset
        else:
            # 没有本地时间戳且无锚点模型时假定已经处于主时基。
            master = _pts_of(packet)
        packet["pts_local"] = local if local is not None else master
        packet["pts"] = master
        packet["timebase"] = "master"
        packet["rate"] = rate
        packet["offset"] = offset
        out.append(packet)

    if state is not None:
        state["timebase"] = "master"
        state.setdefault("models", {}).update(
            {f"{k[0]}:{k[1]}": {"rate": r, "offset": o} for k, (r, o) in models.items()}
        )
    return out


def drift_estimate(anchors: Any, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """根据锚点序列估算每路流相对主时钟的漂移。

    返回每路流的走时速率 ``rate``、相对漂移率 ``drift_ppm``（百万分率）、
    截距 ``offset`` 以及锚点数。漂移率定义为 ``(rate - 1) * 1e6``。
    """
    if not _is_seq(anchors):
        if not isinstance(anchors, dict):
            # 裸标量入口：没有可拟合的锚点，返回零漂移结论而非抛异常。
            result = {
                "streams": {},
                "max_drift_ppm": 0.0,
                "drifting": False,
                "anchor_count": 0,
            }
            if state is not None and isinstance(state, dict):
                state["drift"] = result
            return result
        anchors = [anchors]
    anchor_list = [_normalize_anchor(a) for a in anchors]
    if state is not None:
        anchor_list.extend(_normalize_anchor(a) for a in state.get("anchors", []))

    groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for anchor in anchor_list:
        groups.setdefault((_kind_of(anchor), _source_of(anchor)), []).append(anchor)

    streams: Dict[str, Dict[str, float]] = {}
    rates: List[float] = []
    for (kind, source), group in groups.items():
        group.sort(key=lambda a: a["local"])
        rate, offset = _linear_fit([(a["local"], a["master"]) for a in group])
        rates.append(rate)
        label = f"{kind}:{source}"
        streams[label] = {
            "kind": kind,
            "source": source,
            "rate": rate,
            "offset": offset,
            "drift_ppm": (rate - 1.0) * 1e6,
            "anchor_count": len(group),
        }

    max_abs_ppm = max((abs(s["drift_ppm"]) for s in streams.values()), default=0.0)
    result = {
        "streams": streams,
        "max_drift_ppm": max_abs_ppm,
        "drifting": max_abs_ppm > 1.0,
        "anchor_count": len(anchor_list),
    }
    if state is not None:
        state["drift"] = result
    return result


# --------------------------------------------------------------------------- #
# 2. 空洞插值补齐
# --------------------------------------------------------------------------- #
def gap_fill(packets: Any, nominal_hz: Optional[float] = None,
             max_gap: Optional[float] = None) -> Any:
    """按时间戳检测空洞并线性插值补齐合成包。

    视频丢帧或音频丢包会在时间戳序列上留下空洞（音频本身连续，因此空洞
    一定来自采集丢帧）。以标称间隔（``nominal_hz`` 给出，或由间隔中位数
    推断）为步长，在相邻两个真实包之间插入 ``synthetic=True`` 的补齐包，
    PTS 严格等间隔，数据字段在相邻真实包之间线性插值。

    ``max_gap`` 限制单次补齐的最大时长，超过则视为断流而不无限补包。
    """
    if not _is_seq(packets):
        return packets
    normalized = [_normalize_packet(p) for p in _as_list(packets)]
    if len(normalized) < 2:
        return normalized

    result: List[Dict[str, Any]] = []
    # 同 kind/source 分别处理，避免把两路流之间的间隔误判为空洞。
    groups: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
    for packet in normalized:
        groups.setdefault((_kind_of(packet), _source_of(packet)), []).append(packet)

    emitted: set = set()
    for (kind, source), stream in groups.items():
        stream.sort(key=_pts_of)
        # 优先实测间隔（中位数已剔除空洞、反映漂移校正后的真实帧网格）；
        # 包太少无法实测时才退回标称帧率，避免把大空洞本身当成正常间隔。
        interval = _nominal_interval(stream, None)
        if interval is None and nominal_hz:
            interval = 1.0 / float(nominal_hz)
        if interval is None or interval <= _EPS:
            result.extend(stream)
            continue
        for idx, packet in enumerate(stream):
            result.append(packet)
            emitted.add(id(packet))
            if idx == len(stream) - 1:
                break
            nxt = stream[idx + 1]
            gap = _pts_of(nxt) - _pts_of(packet)
            missing = int(round(gap / interval)) - 1
            if missing <= 0:
                continue
            if max_gap is not None and gap > float(max_gap):
                continue
            for k in range(1, missing + 1):
                ratio = k / (missing + 1)
                synth = {
                    "kind": kind,
                    "source": source,
                    "pts": _pts_of(packet) + interval * k,
                    "pts_local": None,
                    "synthetic": True,
                    "gap_filled": True,
                }
                # 数值型数据字段做线性插值，非数值字段沿用前一个真实包。
                for key, value in packet.items():
                    if key in synth:
                        continue
                    nv = nxt.get(key)
                    if isinstance(value, (int, float)) and isinstance(nv, (int, float)):
                        synth[key] = value + (nv - value) * ratio
                    else:
                        synth[key] = value
                if "frame_no" in packet and "frame_no" in nxt:
                    synth["frame_no"] = packet["frame_no"] + k
                result.append(synth)

    # 保持全局 PTS 单调（跨流合并后的顺序）。
    result.sort(key=lambda p: (_pts_of(p), 0 if not p.get("synthetic") else 1))
    return result


# --------------------------------------------------------------------------- #
# 3. 编码前按 PTS 重排
# --------------------------------------------------------------------------- #
def pts_sort(packets: Any, window: int = DEFAULT_REORDER_WINDOW) -> Any:
    """编码前按 PTS 重排，而不是按到达顺序入队。

    使用有界重排缓冲：每进入一个包就把窗口内最早的包吐出，保证最多容忍
    ``window`` 个包的乱序到达，同时不会为等待迟到包而无限缓存。采用稳定
    排序，PTS 相同的包保持到达顺序（重复时间戳不丢、不乱）。
    """
    if not _is_seq(packets):
        return packets
    window = max(1, int(window))
    buffer: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []
    for index, raw in enumerate(_as_list(packets)):
        packet = _normalize_packet(raw)
        packet.setdefault("arrival_seq", index)
        buffer.append(packet)
        if len(buffer) > window:
            buffer.sort(key=lambda p: (_pts_of(p), p["arrival_seq"]))
            out.append(buffer.pop(0))
    buffer.sort(key=lambda p: (_pts_of(p), p["arrival_seq"]))
    out.extend(buffer)
    return out


# --------------------------------------------------------------------------- #
# 4. 暂停/恢复重置基准
# --------------------------------------------------------------------------- #
def reset_base(state: Any = None, resume_anchor: Any = None,
               master_now: Optional[float] = None) -> Dict[str, Any]:
    """暂停后恢复时重置时钟基准，避免暂停时长被当成时间戳跳变。

    暂停期间设备时钟可能停走或继续走，恢复后必须以恢复瞬间重新建立
    ``local -> master`` 基准：清掉暂停前的模型/锚点，写入恢复锚点，使
    恢复后的包从新基准重新映射（PTS 连续、不包含暂停空档）。

    可传入已有 state（dict）做原地更新并返回；也可独立调用生成新 state。
    """
    if isinstance(state, dict):
        new_state = state
    else:
        new_state = {}

    new_state["paused"] = False
    new_state["resume_count"] = int(new_state.get("resume_count", 0)) + 1
    # 旧基准失效：暂停前的锚点与拟合模型不再适用于恢复后的流。
    new_state["anchors"] = []
    new_state["models"] = {}
    new_state["base_local"] = None
    new_state["base_master"] = None

    if resume_anchor is not None:
        anchor = _normalize_anchor(resume_anchor)
        new_state["anchors"] = [anchor]
        new_state["base_local"] = anchor["local"]
        new_state["base_master"] = anchor["master"]
    elif master_now is not None:
        new_state["base_local"] = 0.0
        new_state["base_master"] = float(master_now)
        new_state["anchors"] = [
            {"local": 0.0, "master": float(master_now), "kind": "video"}
        ]

    new_state["timebase"] = "master"
    return new_state


# --------------------------------------------------------------------------- #
# 5. 热插拔时间戳对齐
# --------------------------------------------------------------------------- #
def hotplug_align(packets: Any, join_anchor: Any = None,
                  source: Any = None) -> Any:
    """设备热插拔后对新流做时间戳对齐。

    新设备的流本地时间戳从零（或任意初值）开始，与主时基没有继承关系。
    ``join_anchor`` 给出新流第一个包对应的 ``(new_local, master)`` 读数，
    本函数据此把新流整体平移（必要时含速率校正）到主时基：

        master = rate * (local - new_local) + master_at_join

    旧流（带主时基 PTS、无本地时间戳）原样透传。
    """
    if not _is_seq(packets):
        return packets
    if join_anchor is None:
        # 缺少接入锚点时无法建立映射，按已在主时基上的流原样返回。
        return [_normalize_packet(p) for p in _as_list(packets)]
    anchor = _normalize_anchor(join_anchor, source=source)
    local0 = anchor["local"]
    master0 = anchor["master"]
    rate = float(anchor.get("rate", 1.0))

    out: List[Dict[str, Any]] = []
    for raw in _as_list(packets):
        packet = _normalize_packet(raw)
        local = _local_of(packet)
        if local is None:
            # 已经在主时基上的旧流，保持不变。
            packet.setdefault("timebase", "master")
            out.append(packet)
            continue
        aligned = master0 + rate * (local - local0)
        packet["pts_local"] = local
        packet["pts"] = aligned
        packet["timebase"] = "master"
        packet["hotplug_aligned"] = True
        if source is not None:
            packet["source"] = source
        out.append(packet)
    return out


# --------------------------------------------------------------------------- #
# 6. 输出前校验
# --------------------------------------------------------------------------- #
def _pair_av(video: Sequence[Dict[str, Any]],
             audio: Sequence[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """按最近 PTS 为每个视频包配对音频包，返回 (video, audio, offset_ms)。"""
    audio_sorted = sorted(audio, key=_pts_of)
    pairs = []
    for v in video:
        vp = _pts_of(v)
        nearest = min(audio_sorted, key=lambda a: abs(_pts_of(a) - vp))
        pairs.append((v, nearest, (_pts_of(nearest) - vp) * 1000.0))
    return pairs


def sync_check(video: Any, audio: Any = None,
               tolerance_ms: float = DEFAULT_AV_TOLERANCE * 1000.0) -> Dict[str, Any]:
    """输出前校验音画偏移。

    对每个视频包找最近音频包，统计音画偏移（音频 PTS - 视频 PTS，毫秒；
    正值表示音频滞后）。返回最大绝对偏移、RMS、是否在容差内及配对明细。
    """
    v_packets = [_normalize_packet(p, "video") for p in _as_list(video)]
    a_packets = [_normalize_packet(p, "audio") for p in _as_list(audio)]
    if not v_packets or not a_packets:
        return {
            "in_sync": True,
            "max_abs_offset_ms": 0.0,
            "rms_offset_ms": 0.0,
            "mean_offset_ms": 0.0,
            "tolerance_ms": float(tolerance_ms),
            "pair_count": 0,
            "pairs": [],
        }

    pairs = _pair_av(v_packets, a_packets)
    offsets = [offset for _v, _a, offset in pairs]
    max_abs = max(abs(o) for o in offsets)
    rms = math.sqrt(sum(o * o for o in offsets) / len(offsets))
    mean = sum(offsets) / len(offsets)
    # 流质心差不受最近邻配对的帧网格量化影响，是整体平移量的无偏估计，
    # 供 fallback 做恒定偏移校正使用。
    centroid = (
        (sum(_pts_of(a) for a in a_packets) / len(a_packets))
        - (sum(_pts_of(v) for v in v_packets) / len(v_packets))
    ) * 1000.0
    return {
        "in_sync": max_abs <= float(tolerance_ms) + _EPS,
        "max_abs_offset_ms": max_abs,
        "rms_offset_ms": rms,
        "mean_offset_ms": mean,
        "centroid_offset_ms": centroid,
        "tolerance_ms": float(tolerance_ms),
        "pair_count": len(pairs),
        "pairs": [
            {
                "video_pts": _pts_of(v),
                "audio_pts": _pts_of(a),
                "offset_ms": offset,
            }
            for v, a, offset in pairs
        ],
    }


def validation(video: Any, audio: Any = None,
               tolerance_ms: float = DEFAULT_AV_TOLERANCE * 1000.0) -> Dict[str, Any]:
    """封装前的综合校验：单调性、空洞、重复 PTS 与音画偏移。"""
    v_packets = [_normalize_packet(p, "video") for p in _as_list(video)]
    a_packets = [_normalize_packet(p, "audio") for p in _as_list(audio)]
    report: Dict[str, Any] = {"streams": {}}

    def _check_stream(name: str, stream: List[Dict[str, Any]]) -> Dict[str, Any]:
        pts_list = [_pts_of(p) for p in stream]
        backwards = sum(
            1 for i in range(1, len(pts_list)) if pts_list[i] < pts_list[i - 1] - _EPS
        )
        unique = set(pts_list)
        duplicates = len(pts_list) - len(unique)
        interval = _nominal_interval(stream, None)
        gaps = 0
        if interval is not None:
            for i in range(1, len(pts_list)):
                if pts_list[i] - pts_list[i - 1] > interval * 1.5:
                    gaps += 1
        return {
            "count": len(stream),
            "backwards": backwards,
            "duplicate_pts": duplicates,
            "gaps": gaps,
            "monotonic": backwards == 0,
            "synthetic": sum(1 for p in stream if p.get("synthetic")),
        }

    report["streams"]["video"] = _check_stream("video", v_packets)
    report["streams"]["audio"] = _check_stream("audio", a_packets)
    av = sync_check(v_packets, a_packets, tolerance_ms)
    report["av"] = {
        "in_sync": av["in_sync"],
        "max_abs_offset_ms": av["max_abs_offset_ms"],
        "rms_offset_ms": av["rms_offset_ms"],
    }
    report["valid"] = (
        report["streams"]["video"]["monotonic"]
        and report["streams"]["audio"]["monotonic"]
        and av["in_sync"]
    )
    return report


def mux_ready(video: Any, audio: Any = None,
              tolerance_ms: float = DEFAULT_AV_TOLERANCE * 1000.0) -> Dict[str, Any]:
    """判断两路流是否达到可封装（mux）条件。"""
    report = validation(video, audio, tolerance_ms)
    ready = (
        report["valid"]
        and report["streams"]["video"]["count"] > 0
        and report["streams"]["audio"]["count"] > 0
    )
    return {"ready": ready, "report": report}


# --------------------------------------------------------------------------- #
# 确定性、审计与兜底
# --------------------------------------------------------------------------- #
def determinism(fn: Any, runs: int = 3, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """验证同一输入多次运行结果完全一致（无随机性、无隐藏状态依赖）。"""
    runs = max(1, int(runs))
    if not callable(fn):
        # 非可调用对象本身即为确定值，直接判定一致。
        return {
            "deterministic": True,
            "runs": runs,
            "output": fn,
        }
    outputs = [fn(*args, **kwargs) for _ in range(runs)]
    first = outputs[0]
    consistent = all(output == first for output in outputs[1:])
    return {
        "deterministic": consistent,
        "runs": runs,
        "output": first,
    }


def audit(video: Any, audio: Any = None,
          state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """生成可追溯的同步质量审计报告。"""
    v_packets = [_normalize_packet(p, "video") for p in _as_list(video)]
    a_packets = [_normalize_packet(p, "audio") for p in _as_list(audio)]
    av = sync_check(v_packets, a_packets)

    def _span(stream: List[Dict[str, Any]]) -> Dict[str, float]:
        if not stream:
            return {"start": 0.0, "end": 0.0, "duration": 0.0}
        pts_list = [_pts_of(p) for p in stream]
        return {
            "start": min(pts_list),
            "end": max(pts_list),
            "duration": max(pts_list) - min(pts_list),
        }

    report = {
        "video": {
            **_span(v_packets),
            "frames": len(v_packets),
            "synthetic_frames": sum(1 for p in v_packets if p.get("synthetic")),
        },
        "audio": {
            **_span(a_packets),
            "packets": len(a_packets),
            "synthetic_packets": sum(1 for p in a_packets if p.get("synthetic")),
        },
        "sync": {
            "in_sync": av["in_sync"],
            "max_abs_offset_ms": av["max_abs_offset_ms"],
            "rms_offset_ms": av["rms_offset_ms"],
            "mean_offset_ms": av["mean_offset_ms"],
        },
    }
    if state is not None:
        report["drift"] = state.get("drift")
        report["resume_count"] = state.get("resume_count", 0)
    return report


def fallback(video: Any, audio: Any = None, strategy: str = "shift",
             tolerance_ms: float = DEFAULT_AV_TOLERANCE * 1000.0) -> Dict[str, Any]:
    """校验失败时的兜底处理：整体平移（shift）或重采样（resample）。

    * ``shift``：保持时长不变，把音频整体平移测得的平均偏移量；
    * ``resample``：按平均偏移在时长上的占比微调整音频 PTS（拉伸/压缩）。
    """
    v_packets = [_normalize_packet(p, "video") for p in _as_list(video)]
    a_packets = [_normalize_packet(p, "audio") for p in _as_list(audio)]
    if not v_packets or not a_packets:
        # 单路流或空输入无法估计偏移，原样返回并标记未做校正。
        return {
            "strategy": strategy,
            "applied": False,
            "correction_ms": 0.0,
            "video": v_packets,
            "audio": a_packets,
            "before": None,
            "after": None,
            "in_sync": True,
        }
    av = sync_check(v_packets, a_packets, tolerance_ms)
    # 用流质心差估计恒定偏移，避免最近邻配对的帧网格量化把平移量估偏。
    mean_offset_sec = av["centroid_offset_ms"] / 1000.0

    fixed_audio: List[Dict[str, Any]] = []
    if a_packets:
        if strategy == "resample":
            a_pts = [_pts_of(p) for p in a_packets]
            span = max(a_pts) - min(a_pts)
            factor = 1.0 - (mean_offset_sec / span) if span > _EPS else 1.0
            base = min(a_pts)
            for packet in a_packets:
                packet = dict(packet)
                packet["pts"] = base + (_pts_of(packet) - base) * factor
                packet["fallback"] = "resample"
                fixed_audio.append(packet)
        else:  # shift（默认）
            for packet in a_packets:
                packet = dict(packet)
                packet["pts"] = _pts_of(packet) - mean_offset_sec
                packet["fallback"] = "shift"
                fixed_audio.append(packet)

    recheck = sync_check(v_packets, fixed_audio, tolerance_ms)
    return {
        "strategy": strategy,
        "applied": not av["in_sync"],
        "correction_ms": mean_offset_sec * 1000.0,
        "video": v_packets,
        "audio": fixed_audio,
        "before": av,
        "after": recheck,
        "in_sync": recheck["in_sync"],
    }


# --------------------------------------------------------------------------- #
# 采集侧（模拟产生带本地时间戳的流）与指标
# --------------------------------------------------------------------------- #
def _emit_stream(kind: str, duration: float, rate_hz: float,
                 clock_rate: float, drop_mask: Iterable[int],
                 source: Any, start_local: float) -> List[Dict[str, Any]]:
    """生成一路带本地时钟时间戳的模拟采集流。

    ``clock_rate`` 模拟设备时钟相对真实时间的走时速率（漂移）；
    ``drop_mask`` 中列出的序号模拟采集丢帧（产生时间戳空洞）。
    """
    count = max(0, int(round(duration * rate_hz)))
    dropped = set(drop_mask or [])
    packets: List[Dict[str, Any]] = []
    for index in range(count):
        if index in dropped:
            continue
        nominal_time = index / rate_hz
        local_ts = start_local + clock_rate * nominal_time
        packet = {
            "kind": kind,
            "source": source,
            "pts_local": local_ts,
            "pts": local_ts,  # 未校正前：直接拿本地时钟当 PTS
            "synthetic": False,
        }
        if kind == "video":
            packet["frame_no"] = index
        else:
            packet["sample_no"] = index
        packets.append(packet)
    return packets


def video_capture(duration: float = 1.0, fps: float = DEFAULT_VIDEO_FPS,
                  clock_rate: float = 1.0, dropped_frames: Optional[Iterable[int]] = None,
                  source: Any = "screen0") -> List[Dict[str, Any]]:
    """模拟屏幕采集线程输出（含本地时钟漂移与丢帧空洞）。"""
    return _emit_stream(
        "video", duration, fps, clock_rate, dropped_frames or [], source, 0.0
    )


def audio_capture(duration: float = 1.0, sample_rate: float = DEFAULT_AUDIO_HZ,
                  clock_rate: float = 1.0, dropped_packets: Optional[Iterable[int]] = None,
                  source: Any = "mic0",
                  start_local: float = 0.0) -> List[Dict[str, Any]]:
    """模拟麦克风采集线程输出（音频连续，默认不丢包）。"""
    return _emit_stream(
        "audio", duration, sample_rate, clock_rate, dropped_packets or [],
        source, start_local,
    )


def metrics(video: Any, audio: Any = None) -> Dict[str, Any]:
    """计算两路流的帧率/丢包/合成占比等采集与同步指标。"""
    v_packets = [_normalize_packet(p, "video") for p in _as_list(video)]
    a_packets = [_normalize_packet(p, "audio") for p in _as_list(audio)]

    def _stream_metrics(stream: List[Dict[str, Any]]) -> Dict[str, float]:
        result = {
            "count": len(stream),
            "synthetic": sum(1 for p in stream if p.get("synthetic")),
            "rate_hz": 0.0,
            "gap_count": 0,
        }
        if len(stream) >= 2:
            pts_list = sorted(_pts_of(p) for p in stream)
            duration = pts_list[-1] - pts_list[0]
            if duration > _EPS:
                result["rate_hz"] = (len(stream) - 1) / duration
            interval = _nominal_interval(stream, None)
            if interval is not None:
                result["gap_count"] = sum(
                    1 for i in range(1, len(pts_list))
                    if pts_list[i] - pts_list[i - 1] > interval * 1.5
                )
        result["synthetic_ratio"] = (
            result["synthetic"] / len(stream) if stream else 0.0
        )
        return result

    av = sync_check(v_packets, a_packets)
    return {
        "video": _stream_metrics(v_packets),
        "audio": _stream_metrics(a_packets),
        "av_max_abs_offset_ms": av["max_abs_offset_ms"],
        "av_rms_offset_ms": av["rms_offset_ms"],
        "in_sync": av["in_sync"],
    }
