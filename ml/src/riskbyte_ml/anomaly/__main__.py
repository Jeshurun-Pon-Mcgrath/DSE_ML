"""Run Phase 5 offline. No command option unlocks the old temporal test partition."""

import argparse
import json
from pathlib import Path

from riskbyte_ml.anomaly.engine import AnomalyConfig
from riskbyte_ml.anomaly.pipeline import run
from riskbyte_ml.anomaly.report import write_reports
from riskbyte_ml.behavioural.deviation import BehaviourConfig
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--behaviour-config", type=Path)
    parser.add_argument(
        "--write-reports", action="store_true", help="Refresh measured Markdown reports"
    )
    args = parser.parse_args()
    try:
        config = load_synthetic_config(args.config)
        anomaly = (
            AnomalyConfig.model_validate_json(args.experiment_config.read_bytes())
            if (args.experiment_config)
            else AnomalyConfig()
        )
        behaviour = (
            BehaviourConfig.model_validate_json(args.behaviour_config.read_bytes())
            if (args.behaviour_config)
            else BehaviourConfig()
        )
        root = Path(__file__).resolve().parents[3] / "artifacts"
        new_run = not (root / "phase5" / "analysis_bundle.json").exists()
        result = run(config, root, anomaly, behaviour)
        if new_run or args.write_reports:
            write_reports(result, root.parents[1] / "docs")
        print(
            json.dumps(
                {
                    "dataset_fingerprint": result.dataset_fingerprint,
                    "anomaly_version": result.anomaly_version,
                    "behaviour_version": result.behaviour_version,
                    "selected_family": result.selected["family"],
                    "contamination": result.selected["contamination"],
                    "anomaly_validation": result.anomaly_metrics,
                    "validation_complementarity": result.complementarity["validation"],
                    "leakage_audit": result.leakage_audit,
                },
                indent=2,
                allow_nan=False,
            )
        )
    except (DatasetError, OSError, ValueError) as exc:
        print(f"Phase 5 aborted: {getattr(exc, 'message', 'Invalid configuration or data')}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
