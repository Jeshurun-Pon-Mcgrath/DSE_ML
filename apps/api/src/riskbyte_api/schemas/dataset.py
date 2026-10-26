"""Strict, versioned data intelligence response envelopes; no Pandas or local paths."""

from datetime import datetime
from typing import Generic, Literal, TypeVar

from pydantic import model_serializer, model_validator

from riskbyte_ml.data.models import ReportModel

T = TypeVar("T")
type DatasetState = Literal["dataset_ready", "dataset_validation_failed", "dataset_not_configured"]


class DatasetStatus(ReportModel):
    dataset_name: str
    fingerprint: str
    generated_at: datetime
    row_count: int
    column_count: int
    validation_status: Literal["PASSED", "FAILED"]


class DatasetResponse(ReportModel, Generic[T]):
    status: DatasetState
    data: T | None = None

    @model_serializer(mode="wrap")
    def serialize_response(self, handler):
        result = handler(self)
        if self.status == "dataset_not_configured":
            result.pop("data", None)
        return result

    @model_validator(mode="after")
    def consistent_state(self):
        if (self.status == "dataset_not_configured") != (self.data is None):
            raise ValueError(
                "Unconfigured datasets have no report; configured datasets require data"
            )
        return self
