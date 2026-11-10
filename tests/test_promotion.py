import json
from pathlib import Path

import dagshub
import mlflow

from evals.mlflow_utils import EXPERIMENT_NAME, get_run_by_stage

THRESHOLDS_PATH = Path(__file__).parent.parent / "evals" / "thresholds.json"
NOISE_MULTIPLIER = 2
MIN_IMPROVED_METRICS = 1


dagshub.init(repo_owner="k-Rohit", repo_name="PaperPilot", mlflow=True)
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
champion = get_run_by_stage(experiment_id, "champion").data.metrics
challenger = get_run_by_stage(experiment_id, "challenger").data.metrics
noise = json.loads(THRESHOLDS_PATH.read_text())["noise_thresholds"]


def test_at_least_one_real_improvement():
    real_gains = {}
    for metric in noise:
        tolerance = NOISE_MULTIPLIER * noise[metric]
        delta = challenger[metric] - champion[metric]
        if delta > tolerance:
            real_gains[metric] = delta
    assert len(real_gains) >= MIN_IMPROVED_METRICS, (
        f"Only {len(real_gains)} metric(s) improved beyond noise, need {MIN_IMPROVED_METRICS}: {real_gains}"
    )
