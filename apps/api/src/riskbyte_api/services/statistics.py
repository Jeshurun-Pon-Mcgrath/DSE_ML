"""Serve a coherent offline statistical snapshot after exact provenance verification."""

from pathlib import Path

from riskbyte_api.core.config import REPOSITORY_ROOT, Settings
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.statistics import (
    ARTIFACT_VERSION,
    FROZEN_FEATURE_VERSION,
    FROZEN_FINGERPRINT,
    FROZEN_SPLIT_ID,
)
from riskbyte_ml.statistics.models import StatisticalBundle, StatisticsResponse, StatisticsSummary
from riskbyte_ml.statistics.verification import frozen_artifacts, verify_sources


class StatisticsService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def resolve(path: Path):
        return path if path.is_absolute() else REPOSITORY_ROOT / path

    def response(self, section):
        path = self.settings.dataset_config_path
        if path is None:
            return StatisticsResponse(status="dataset_not_configured")
        if self.settings.dataset_adapter != "synthetic_riskbyte":
            return StatisticsResponse(
                status="statistics_unavailable",
                reason="Phase 3 statistics require the frozen RiskByte synthetic dataset",
            )
        config = load_synthetic_config(self.resolve(path))
        verify_sources(config)  # Content hashes, not filenames or file-attribute cache guesses.
        directory = self.resolve(self.settings.statistics_artifacts_dir)
        frozen_artifacts(directory.parent)
        bundle_path = directory / "analysis_bundle.json"
        if not bundle_path.is_file():
            return StatisticsResponse(status="statistics_not_generated")
        try:
            bundle = (
                StatisticalBundle.model_validate_json(bundle_path.read_bytes())
                if (bundle_path.stat().st_size <= 10_000_000)
                else None
            )
            if bundle is None:
                raise ValueError
            provenance = bundle.provenance
            if (
                provenance.dataset_fingerprint != FROZEN_FINGERPRINT
                or provenance.feature_pipeline_version != FROZEN_FEATURE_VERSION
                or provenance.split_id != FROZEN_SPLIT_ID
                or provenance.artifact_version != ARTIFACT_VERSION
            ):
                raise ValueError
        except (OSError, ValueError):
            raise DatasetError(
                "statistics_bundle_invalid", "Statistical snapshot is invalid or incompatible"
            ) from None
        if section == "summary":
            data = StatisticsSummary(
                provenance=provenance,
                descriptive=bundle.descriptive,
                hypothesis_count=bundle.hypotheses.family_size,
                nominal_rejections=sum(
                    h.decision == "REJECT_H0_NOMINALLY" for h in bundle.hypotheses.hypotheses
                ),
            )
        else:
            data = getattr(bundle, section)
        return StatisticsResponse(status="statistics_ready", data=data, provenance=provenance)
