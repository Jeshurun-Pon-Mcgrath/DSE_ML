import argparse
import json
from pathlib import Path

from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.final_holdout.pipeline import evaluate_once, prepare


def main():
    parser = argparse.ArgumentParser(description="One-use frozen future holdout")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = load_synthetic_config(args.config)
    result = (
        prepare(config, Path("ml/artifacts"))[0]
        if args.prepare_only
        else evaluate_once(config, Path("ml/artifacts"))
    )
    print(json.dumps({"status": result["status"], "fingerprint": result["holdout_fingerprint"]}))


if __name__ == "__main__":
    main()
