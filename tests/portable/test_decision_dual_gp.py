import pytest

from c2kv_eval.portable.decision_dual_gp import DecisionDualGPScheduler


def gp(mean):
    return {
        "scaler_mean": [0.0] * 4,
        "scaler_scale": [1.0] * 4,
        "x_train_scaled": [[0.0] * 4],
        "alpha": [0.0],
        "cholesky": [[1.0]],
        "constant": 1.0,
        "length_scale": [1.0] * 4,
        "noise": 0.0,
        "y_mean": mean,
        "y_std": 0.0,
    }


def artifact(risk=0.1):
    tasks = [f"multi_turn_long_context_{index}" for index in range(200)]
    mapping = {task: index % 5 for index, task in enumerate(tasks)}
    return {
        "schema": "bfcl-decision-dual-gp-v1",
        "frozen": True,
        "task_to_fold": mapping,
        "fold_models": {
            str(fold): {
                "gain_gp": gp(100.0),
                "risk_gp": gp(risk),
                "risk_threshold": 0.25,
                "uncertainty_beta": 1.0,
                "training_tasks": [
                    task for task in tasks if mapping[task] != fold
                ],
            }
            for fold in range(5)
        },
    }


def observed(history=1000):
    return {
        "L_total": 4000.0,
        "L_hist": float(history),
        "L_tool": 2000.0,
    }


def test_selects_racer_with_positive_gain_and_low_risk():
    scheduler = DecisionDualGPScheduler(artifact())
    result = scheduler.route_observed("multi_turn_long_context_0", observed())
    assert result["selected_arm"] == "c2kv_racer"
    assert result["held_out_fold"] == 0


def test_no_history_and_high_risk_fail_closed():
    scheduler = DecisionDualGPScheduler(artifact())
    assert scheduler.route_observed(
        "multi_turn_long_context_1", observed(0)
    )["selected_arm"] == "full"
    risky = DecisionDualGPScheduler(artifact(risk=0.4))
    assert risky.route_observed(
        "multi_turn_long_context_1", observed()
    )["selected_arm"] == "full"


def test_fold_leakage_is_rejected():
    payload = artifact()
    payload["fold_models"]["0"]["training_tasks"].append(
        "multi_turn_long_context_0"
    )
    scheduler = DecisionDualGPScheduler(payload)
    with pytest.raises(AssertionError, match="fold leakage"):
        scheduler.route_observed("multi_turn_long_context_0", observed())
