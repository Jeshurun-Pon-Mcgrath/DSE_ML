"""Explicit offline Phase 4; completed experiments are reused without reopening test data."""

import argparse
import json
from pathlib import Path

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.models.supervised.pipeline import run_experiment
from riskbyte_ml.models.supervised.schemas import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Frozen RiskByte supervised classification")
    parser.add_argument(
        "--config", type=Path, required=True, help="Existing synthetic adapter JSON"
    )
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=Path("ml/artifacts"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    try:
        policy = (
            ExperimentConfig.model_validate_json(args.experiment_config.read_bytes())
            if args.experiment_config
            else ExperimentConfig()
        )
        report = run_experiment(
            load_synthetic_config(args.config), args.artifact_root, policy, args.output_root
        )
    except (DatasetError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "training_aborted",
                    "code": exc.code
                    if isinstance(exc, DatasetError)
                    else "model_configuration_invalid",
                    "message": exc.message
                    if isinstance(exc, DatasetError)
                    else "Invalid experiment configuration",
                }
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "models_ready",
                "model_version": report.provenance.model_version,
                "selected_model": report.selection_freeze.model_id,
                "run_id": report.provenance.run_id,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
