"""Host-wall first-sampled-token timing, independent of HTTP buffering."""

import math


def restore_model_first_token_time(output, index):
    """Restore scheduler timestamps from their explicit batch transport."""
    first_stamps = getattr(output, "model_first_token_times", None)
    forward_stamps = getattr(output, "model_forward_entry_times", None)
    stats = getattr(output, "time_stats", None)
    if (
        not isinstance(index, int)
        or index < 0
        or first_stamps is None
        or forward_stamps is None
        or stats is None
        or index >= len(first_stamps)
        or index >= len(forward_stamps)
        or index >= len(stats)
        or stats[index] is None
    ):
        return False
    first = first_stamps[index]
    forward = forward_stamps[index]
    if (
        not isinstance(first, (int, float))
        or not math.isfinite(first)
        or first <= 0
        or not isinstance(forward, (int, float))
        or not math.isfinite(forward)
        or forward <= 0
    ):
        return False
    stats[index].model_first_token_time = float(first)
    stats[index].forward_entry_time = float(forward)
    return True


def timing_metadata(api, scheduler):
    if scheduler is None:
        return {}
    first = getattr(scheduler, "model_first_token_time", 0.0)
    start = api.created_time
    forward = scheduler.forward_entry_time
    if not all(math.isfinite(t) and t > 0 for t in (first, start, forward)):
        return {}
    if first < max(start, forward):
        return {}
    return {
        "schema": "native-speed-timing-v1",
        "clock": "perf_counter",
        "first_token_source": "scheduler_first_sampled_output_before_transport",
        "server_ttft_ms": (first - start) * 1000,
        "prefill_path_ms": (first - forward) * 1000,
        "scope": "Host wall including KV injection/inter-round waits; not CUDA kernel time",
    }


def attach_timing_metadata(meta_info, api, scheduler):
    """Attach native first-token timing independently of metrics export."""
    speed = timing_metadata(api, scheduler)
    if not speed:
        return False
    meta_info["speed_timing"] = speed
    return True
