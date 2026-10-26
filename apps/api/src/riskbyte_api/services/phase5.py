"""Read-only Phase 5 reports. Never fit, unpickle, or score in an HTTP request."""

from pathlib import Path

from riskbyte_api.core.config import REPOSITORY_ROOT
from riskbyte_ml.anomaly.engine import ANOMALY_VERSION
from riskbyte_ml.anomaly.schemas import Phase5Bundle, Phase5Response
from riskbyte_ml.behavioural.deviation import BEHAVIOUR_VERSION
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.models.supervised.inputs import verify_frozen_files
from riskbyte_ml.statistics import FROZEN_FEATURE_VERSION, FROZEN_FINGERPRINT, FROZEN_SPLIT_ID
from riskbyte_ml.statistics.verification import file_hash, verify_sources


class Phase5Service:
    def __init__(self, settings):
        self.settings = settings

    def response(self, section):
        settings = self.settings
        if settings.dataset_config_path is None:
            return Phase5Response(status="dataset_not_configured")
        if settings.dataset_adapter != "synthetic_riskbyte":
            return Phase5Response(
                status="phase5_unavailable", reason="Frozen synthetic dataset required"
            )
        source = settings.dataset_config_path
        source = source if source.is_absolute() else REPOSITORY_ROOT / source
        config = load_synthetic_config(source)
        verify_sources(config)
        directory = settings.phase5_artifacts_dir
        directory = directory if directory.is_absolute() else REPOSITORY_ROOT / directory
        verify_frozen_files(directory.parent)
        path = directory / "analysis_bundle.json"
        if not path.is_file():
            return Phase5Response(status="phase5_not_generated")
        try:
            if path.stat().st_size > 5_000_000:
                raise ValueError
            bundle = Phase5Bundle.model_validate_json(path.read_bytes())
            if (
                bundle.dataset_fingerprint != FROZEN_FINGERPRINT
                or bundle.feature_version != FROZEN_FEATURE_VERSION
                or bundle.split_id != FROZEN_SPLIT_ID
                or bundle.anomaly_version != ANOMALY_VERSION
                or bundle.behaviour_version != BEHAVIOUR_VERSION
                or bundle.leakage_audit.get("original_temporal_test_accessed") is not False
            ):
                raise ValueError
            for name, expected in bundle.artifact_hashes.items():
                if Path(name).name != name or file_hash(directory / name) != expected:
                    raise ValueError
            if (
                file_hash(directory.parent / "models" / "model_registry.json")
                != (bundle.baseline["registry_hash"])
            ):
                raise ValueError
        except (OSError, ValueError, KeyError):
            raise DatasetError("phase5_artifact_invalid", "Phase 5 snapshot is invalid") from None
        sections = {
            "status": {
                "dataset_fingerprint": bundle.dataset_fingerprint,
                "feature_version": bundle.feature_version,
                "anomaly_version": bundle.anomaly_version,
                "behaviour_version": bundle.behaviour_version,
                "scope": "train and validation only",
            },
            "metrics": bundle.anomaly_metrics,
            "scenarios": {
                "anomaly": bundle.anomaly_metrics["scenario_recall"],
                "supervised": bundle.complementarity["supervised_validation"]["scenario_recall"],
                "behaviour": bundle.distributions["scenarios"],
            },
            "hard-negatives": {
                "anomaly": bundle.anomaly_metrics["hard_negative_types"],
                "behaviour": bundle.behaviour_metrics["hard_negative_types"],
                "score_distributions": bundle.distributions["hard_negative_types"],
            },
            "summary": {
                "version": bundle.behaviour_version,
                "groups": bundle.distributions["groups"],
                "score_semantics": "deviation, not fraud probability",
            },
            "components": bundle.behavioural_components,
            "validation-analysis": {
                "behaviour": bundle.behaviour_metrics,
                "complementarity": bundle.complementarity["validation"],
                "investigations": bundle.investigations,
            },
        }
        return Phase5Response(status="phase5_ready", data=sections[section])
