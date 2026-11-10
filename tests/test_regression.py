import json
from pathlib import Path

import dagshub
import mlflow
import pytest

from evals.mlflow_utils import EXPERIMENT_NAME, get_run_by_stage

THRESHOLDS_PATH = Path(__file__).parent.parent / "evals" / "thresholds.json"
NOISE_MULTIPLIER = 2


dagshub.init(repo_owner="k-Rohit", repo_name="PaperPilot", mlflow=True)
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
champion = get_run_by_stage(experiment_id, "champion").data.metrics
challenger = get_run_by_stage(experiment_id, "challenger").data.metrics
noise = json.loads(THRESHOLDS_PATH.read_text())["noise_thresholds"]


@pytest.mark.parametrize("metric", sorted(noise))
def test_no_regression(metric: str):
    tolerance = NOISE_MULTIPLIER * noise[metric]
    lower_bound = champion[metric] - tolerance
    assert challenger[metric] >= lower_bound, (
        f"{metric} regressed: challenger={challenger[metric]:.3f} < "
        f"champion={champion[metric]:.3f} - tolerance={tolerance:.3f} = {lower_bound:.3f}"
    )
