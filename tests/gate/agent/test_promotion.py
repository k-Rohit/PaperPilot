from evals.mlflow_utils import AGENT_EXPERIMENT_NAME, PIPELINES, load_gate_inputs, load_noise_floors, tolerance

_, THRESHOLDS_PATH = PIPELINES["agent"]
MIN_IMPROVED_METRICS = 1

champion, challenger, noise = load_gate_inputs(AGENT_EXPERIMENT_NAME, THRESHOLDS_PATH)
floors = load_noise_floors(THRESHOLDS_PATH)


def test_at_least_one_real_improvement():
    real_gains = {}
    for metric in noise:
        delta = challenger[metric] - champion[metric]
        if delta > tolerance(metric, noise, floors):
            real_gains[metric] = delta
    assert len(real_gains) >= MIN_IMPROVED_METRICS, (
        f"Only {len(real_gains)} metric(s) improved beyond noise, need {MIN_IMPROVED_METRICS}: {real_gains}"
    )
