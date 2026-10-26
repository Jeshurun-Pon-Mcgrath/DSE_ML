"""Generate immutable offline Phase 7 artifacts without scoring original/future holdouts."""

import argparse
import json
from pathlib import Path

from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.explainability.pipeline import run
from riskbyte_ml.explainability.schemas import SHAP_VERSION, VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, default=Path("ml/artifacts"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=5, choices=range(1, 21))
    parser.add_argument("--snapshot-version", choices=[VERSION, SHAP_VERSION], default=VERSION)
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    try:
        bundle = run(
            load_synthetic_config(args.config),
            args.artifacts_root,
            args.output_dir,
            args.top_k,
            args.snapshot_version,
            args.baseline_dir,
        )
    except DatasetError as exc:
        parser.exit(1, f"{exc.code}: {exc}\n")
    print(
        json.dumps(
            {
                "version": bundle.version,
                "shap_status": bundle.shap_status,
                "dataset_fingerprint": bundle.provenance["dataset_fingerprint"],
                "original_temporal_test_accessed": False,
                "future_holdout_status": bundle.future_holdout_status,
            }
        )
    )


if __name__ == "__main__":
    main()
