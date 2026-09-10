"""
Gate A run_003 sign-audit: verifies the response_delay_frames convention against a
deterministic synthetic sequence where action is KNOWN to lead state by +3 frames.

Convention under test:
  response_delay_frames D (found by minimizing MAE(action[t], state[t+D]) over t):
    D > 0  => state[t] reflects action issued D frames earlier => ACTION LEADS STATE by D.
    D < 0  => action[t] matches an earlier state value => ACTION LAGS STATE (state leads).
    D == 0 => synchronous / no measurable delay.

This corrects a run_001/run_002 bug: the scan code used there paired action[t] with
state[t - lag] while lag<0 was described in prose as "state[t+lag]" -- the code and the
prose used opposite signs of the same slice, so the *numeric* best-lag values from
run_001/run_002 (-2, -4, -2, -4) were mislabeled. Under the corrected, sign-tested
convention below, those same 4 fields recover as response_delay_frames = +2, +4, +2, +4
(action leads), which is re-derived from scratch (not by relabeling old numbers) in
core_delay_scan_corrected_110ep_final.json.
"""
import numpy as np


def verified_scan(action, state, delays):
    """response_delay_frames D: compare action[t] vs state[t+D].
    D >= 0: pair action[0 : n-D] with state[D : n]      (state sampled D steps AHEAD in index)
    D < 0:  pair action[-D : n] with state[0 : n+D]      (state sampled |D| steps BEHIND in index)
    """
    n = len(action)
    res = {}
    for D in delays:
        if D >= 0:
            a = action[: n - D] if D > 0 else action
            s = state[D:]
        else:
            a = action[-D:]
            s = state[: n + D]
        m = min(len(a), len(s))
        res[D] = float(np.mean(np.abs(a[:m] - s[:m])))
    return res


def build_synthetic(true_delay, n=200, seed=0):
    """Builds action_full, then state[t] = action[t - true_delay] for t >= true_delay
    (state echoes a past action => action genuinely leads state by true_delay frames).
    Returns (action, state) restricted to the valid overlapping window, both length n."""
    rng = np.random.default_rng(seed)
    N = n + true_delay
    action_full = np.sin(np.linspace(0, 20, N)) + 0.01 * rng.standard_normal(N)
    state_full = np.full(N, np.nan)
    state_full[true_delay:] = action_full[: N - true_delay]
    valid = slice(true_delay, N)
    action = action_full[valid]
    state = state_full[valid]
    assert not np.isnan(state).any()
    # sanity: state[t] == action_full[t - true_delay] for the retained window
    t_check = np.arange(true_delay, true_delay + 5)
    assert np.allclose(state_full[t_check], action_full[t_check - true_delay])
    return action, state


def run_sign_test(true_delay=3, delays=range(-6, 7)):
    action, state = build_synthetic(true_delay)
    res = verified_scan(action, state, list(delays))
    best = min(res, key=res.get)
    assert best == true_delay, (
        f"SIGN TEST FAILED: injected action-leads-state delay={true_delay}, "
        f"recovered best response_delay_frames={best}. "
        f"Convention is inverted -- do not trust any delay/lag conclusion until fixed."
    )
    return {"true_delay": true_delay, "recovered_delay": best, "mae_by_delay": res, "passed": True}


if __name__ == "__main__":
    import json
    result = run_sign_test(true_delay=3, delays=range(-6, 7))
    print(json.dumps(result, indent=2))
    print("SIGN TEST PASSED: response_delay_frames convention verified.")
