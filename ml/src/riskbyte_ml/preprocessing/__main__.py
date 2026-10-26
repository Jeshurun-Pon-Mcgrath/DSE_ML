"""Reproduce Phase 2 artifacts from existing fingerprint-verified synthetic sources."""

import argparse
import json
from pathlib import Path

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.datasets.synthetic_riskbyte.pipeline import run_phase2
from riskbyte_ml.evaluation.splits import SplitConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="RiskByte Phase 2; preprocessing only, no models")
    parser.add_argument(
        "--config", type=Path, default=Path("ml/config/synthetic_riskbyte.example.json")
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("ml/artifacts"))
    parser.add_argument(
        "--split-strategy", choices=["temporal", "random_stratified"], default="temporal"
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        summary = run_phase2(
            load_synthetic_config(args.config),
            args.artifacts_dir,
            split_config=SplitConfig(
                strategy=args.split_strategy,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                seed=args.seed,
            ),
        )
        print(
            json.dumps(
                {
                    "status": "preprocessing_ready",
                    "dataset_fingerprint": summary["dataset"]["dataset_fingerprint"],
                    "split_strategy": summary["split_strategy"],
                    "features": summary["features"],
                }
            )
        )
        return 0
    except (DatasetError, ValueError, OSError) as error:
        print(
            json.dumps(
                {
                    "status": "dataset_configuration_error",
                    "error": {
                        "code": error.code
                        if isinstance(error, DatasetError)
                        else "preprocessing_failed",
                        "message": error.message
                        if isinstance(error, DatasetError)
                        else "Preprocessing failed; review local configuration",
                    },
                }
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
