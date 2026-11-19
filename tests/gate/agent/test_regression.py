import pytest

from evals.mlflow_utils import AGENT_EXPERIMENT_NAME, PIPELINES, load_gate_inputs, load_noise_floors, tolerance

_, THRESHOLDS_PATH = PIPELINES["agent"]

champion, challenger, noise = load_gate_inputs(AGENT_EXPERIMENT_NAME, THRESHOLDS_PATH)
floors = load_noise_floors(THRESHOLDS_PATH)


@pytest.mark.parametrize("metric", sorted(noise))
def test_no_regression(metric: str):
    tol = tolerance(metric, noise, floors)
    lower_bound = champion[metric] - tol
    assert challenger[metric] >= lower_bound, (
        f"{metric} regressed: challenger={challenger[metric]:.3f} < "
        f"champion={champion[metric]:.3f} - tolerance={tol:.3f} = {lower_bound:.3f}"
    )
