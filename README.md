# pairwise-capture-av-sync-drift

## 领域

游戏、图形与多媒体

## 主题

采集音画同步漂移

## 说明

统一时基、空洞插值与热插拔时间戳对齐。

初始状态下 `media/capture_av_sync_drift.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
