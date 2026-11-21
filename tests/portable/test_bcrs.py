from c2kv_eval.portable.bcrs import gain_probability, prior_guided_decision


RULE = {
    "history_length_prior_tokens": 20000,
    "maximum_semantic_risk": 0.65,
    "semantic_risk_weights": {"s_tool": 0.2, "s_hist": 0.45, "d_cross": 0.35},
    "v2_full_below_tokens": 16000,
    "v2_racer_at_or_above_tokens": 24000,
    "v2_high_gain_probability": 0.95,
    "v2_low_gain_probability": 0.05,
}


def observed(total, history=1000, risk=0.0):
    return {
        "L_total": total,
        "L_tool": max(0, total - history),
        "L_hist": history,
        "rho_tool": 0.5,
        "S_tool": risk,
        "S_hist": risk,
        "D_cross": risk,
    }


def test_gain_probability_uses_log_ratio_sign():
    assert gain_probability(-1.0, 0.1) > 0.99
    assert gain_probability(1.0, 0.1) < 0.01


def test_length_priors_and_uncertain_fixed20k_fallback():
    assert not prior_guided_decision(observed(15999), -10.0, 0.1, RULE)["compress"]
    assert prior_guided_decision(observed(24000), 10.0, 0.1, RULE)["compress"]
    assert not prior_guided_decision(observed(19000), 0.0, 1.0, RULE)["compress"]
    assert prior_guided_decision(observed(21000), 0.0, 1.0, RULE)["compress"]


def test_boundary_high_confidence_can_modify_prior():
    assert prior_guided_decision(observed(18000), -1.0, 0.1, RULE)["compress"]
    assert not prior_guided_decision(observed(22000), 1.0, 0.1, RULE)["compress"]


def test_explicit_semantic_risk_and_empty_history_fail_closed():
    assert not prior_guided_decision(observed(30000, risk=1.0), -1.0, 0.1, RULE)["compress"]
    assert not prior_guided_decision(observed(30000, history=0), -1.0, 0.1, RULE)["compress"]
