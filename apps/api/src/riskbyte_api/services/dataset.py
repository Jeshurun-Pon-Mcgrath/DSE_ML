"""Lazy, bounded per-process snapshots; concurrent requests share one computation."""

from threading import Lock

from fastapi import Request

from riskbyte_api.core.config import REPOSITORY_ROOT, Settings
from riskbyte_ml.data.config import load_config
from riskbyte_ml.data.models import DatasetProfile
from riskbyte_ml.data.pipeline import profile_dataset, save_reports
from riskbyte_ml.datasets.ieee_cis.config import load_ieee_config
from riskbyte_ml.datasets.ieee_cis.pipeline import integrate, save_integration
from riskbyte_ml.datasets.synthetic_riskbyte.config import load_synthetic_config
from riskbyte_ml.datasets.synthetic_riskbyte.pipeline import (
    integrate as integrate_synthetic,
)
from riskbyte_ml.datasets.synthetic_riskbyte.pipeline import (
    save_integration as save_synthetic_integration,
)


class DatasetService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._snapshot: DatasetProfile | None = None
        self._key = None
        self._integration = None

    def snapshot(self) -> DatasetProfile | None:
        with self._lock:
            path = self.settings.dataset_config_path
            if path is None:
                return None
            if not path.is_absolute():
                path = REPOSITORY_ROOT / path
            ieee = self.settings.dataset_adapter == "ieee_cis"
            synthetic = self.settings.dataset_adapter == "synthetic_riskbyte"
            config = (
                load_synthetic_config(path)
                if synthetic
                else load_ieee_config(path)
                if ieee
                else load_config(path)
            )
            if not ieee and not synthetic and config.source_path is None:
                self._snapshot, self._key = None, None
                self._integration = None
                return None
            sources = (
                (config.transaction_path, config.identity_path, config.metadata_path)
                if synthetic
                else (config.transaction_path, config.identity_path)
                if ieee
                else (config.source_path,)
            )
            file_keys = []
            if synthetic and config.phase2_summary_path is not None:
                sources = (*sources, config.phase2_summary_path)
            for source in sources:
                try:
                    stat = source.stat()
                    file_keys.append((stat.st_size, stat.st_mtime_ns, stat.st_ino))
                except OSError:
                    file_keys.append(None)  # The adapter emits sanitized domain errors.
            key = (self.settings.dataset_adapter, config.model_dump_json(), tuple(file_keys))
            if self._snapshot is None or key != self._key:
                integration = None
                if synthetic:
                    local_profile, integration = integrate_synthetic(config)
                elif ieee:
                    local_profile, integration = integrate(config)
                else:
                    local_profile = profile_dataset(config)
                directory = self.settings.dataset_artifacts_dir
                if not directory.is_absolute():
                    directory = REPOSITORY_ROOT / directory
                if synthetic:
                    save_synthetic_integration(local_profile, integration, directory, config)
                elif ieee:
                    save_integration(local_profile, integration, directory, config)
                else:
                    save_reports(local_profile, directory)
                self._snapshot = DatasetProfile.model_validate(
                    local_profile.model_dump(exclude={"source_path"})
                )
                self._key = key
                self._integration = integration
            return self._snapshot

    def integration_snapshot(self):
        # Obtain profile + adapter metadata from the same cached generation.
        while True:
            profile = self.snapshot()
            with self._lock:
                if profile is None or profile is self._snapshot:
                    return profile, self._integration if profile is not None else None


def get_dataset_service(request: Request) -> DatasetService:
    return request.app.state.dataset_service
