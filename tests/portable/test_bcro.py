import math

from c2kv_eval.portable.bcro import bcro_decision


RULE = {
    "uncertainty_beta": 1.0,
    "semantic_risk_lambda": 0.35,
    "maximum_semantic_risk": 0.65,
    "semantic_risk_weights": {"s_tool": 0.20, "s_hist": 0.45, "d_cross": 0.35},
}


def observed(total=8000, history=1000, risk=0.0):
    return {
        "L_total": total,
        "L_tool": total - history,
        "L_hist": history,
        "rho_tool": (total - history) / total,
        "S_tool": risk,
        "S_hist": risk,
        "D_cross": risk,
    }


def test_bcro_has_no_length_threshold():
    short_gain = bcro_decision(observed(4000), 100.0, 40.0, 0.0, 0.01, RULE)
    long_loss = bcro_decision(observed(64000), 100.0, 140.0, 0.0, 0.01, RULE)
    assert short_gain["compress"]
    assert not long_loss["compress"]


def test_bcro_quality_guards_are_fail_closed():
    assert not bcro_decision(observed(history=0), 100, 20, 0, 0.1, RULE)["compress"]
    risky = bcro_decision(observed(risk=1.0), 100, 20, 0, 0.1, RULE)
    assert not risky["compress"]
    assert risky["reason"] == "semantic_risk_guard"
    invalid = bcro_decision(observed(), math.nan, 20, 0, 0.1, RULE)
    assert not invalid["compress"]
    assert invalid["reason"] == "predictor_invalid_full_fallback"
