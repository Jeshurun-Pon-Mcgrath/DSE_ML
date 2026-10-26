"""Read-only data intelligence endpoints; operators configure datasets server-side."""

from typing import Annotated

from fastapi import APIRouter, Depends

from riskbyte_api.schemas.dataset import DatasetResponse, DatasetStatus
from riskbyte_api.services.dataset import DatasetService, get_dataset_service
from riskbyte_ml.data.models import (
    Analysis,
    DatasetProfile,
    EDAReport,
    NumericStatistics,
    QualityReport,
    SchemaReport,
)
from riskbyte_ml.datasets.ieee_cis.models import IntegrationReport
from riskbyte_ml.datasets.synthetic_riskbyte.models import SyntheticMetadata

router = APIRouter(prefix="/api/v1/dataset", tags=["dataset"])
Service = Annotated[DatasetService, Depends(get_dataset_service)]


def response(report, data):
    if report is None:
        return {"status": "dataset_not_configured"}
    return {
        "status": "dataset_ready"
        if report.quality.validation_status == "PASSED"
        else "dataset_validation_failed",
        "data": data,
    }


@router.get("/status", response_model=DatasetResponse[DatasetStatus])
def dataset_status(service: Service):
    report = service.snapshot()
    status = (
        None
        if report is None
        else DatasetStatus(
            dataset_name=report.dataset_name,
            fingerprint=report.fingerprint,
            generated_at=report.generated_at,
            row_count=report.row_count,
            column_count=report.column_count,
            validation_status=report.quality.validation_status,
        )
    )
    return response(report, status)


@router.get("/profile", response_model=DatasetResponse[DatasetProfile])
def dataset_profile(service: Service):
    report = service.snapshot()
    return response(report, report)


@router.get("/quality", response_model=DatasetResponse[QualityReport])
def dataset_quality(service: Service):
    report = service.snapshot()
    return response(report, report.quality if report else None)


@router.get(
    "/statistics",
    response_model=DatasetResponse[dict[str, NumericStatistics]],
)
def dataset_statistics(service: Service):
    report = service.snapshot()
    return response(report, report.descriptive_statistics if report else None)


@router.get("/eda", response_model=DatasetResponse[EDAReport])
def dataset_eda(service: Service):
    report = service.snapshot()
    return response(report, report.eda if report else None)


@router.get("/schema", response_model=DatasetResponse[SchemaReport])
def dataset_schema(service: Service):
    report = service.snapshot()
    return response(report, report.schema_metadata if report else None)


@router.get(
    "/integration", response_model=DatasetResponse[Analysis[IntegrationReport | SyntheticMetadata]]
)
def dataset_integration(service: Service):
    profile, integration = service.integration_snapshot()
    analysis = Analysis[IntegrationReport | SyntheticMetadata](
        status="AVAILABLE" if integration else "UNAVAILABLE_WITH_REASON",
        data=integration,
        reason=None if integration else "dataset_adapter_is_not_ieee_cis",
    )
    return response(profile, analysis if profile else None)
