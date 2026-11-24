import pytest
import media.capture_av_sync_drift as mod

CASES = [
    ("unify_timebase", (1,)),
    ("drift_estimate", (1,)),
    ("gap_fill", (1,)),
    ("pts_sort", (1,)),
    ("reset_base", (1,)),
    ("hotplug_align", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("video_capture", (1,)),
    ("metrics", (1,)),
    ("audio_capture", (1,)),
    ("mux_ready", (1,)),
    ("sync_check", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
