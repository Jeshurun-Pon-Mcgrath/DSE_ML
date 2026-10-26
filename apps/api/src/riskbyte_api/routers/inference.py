"""Frontend-facing frozen inference, demo browsing and final evaluation."""

from datetime import UTC
from secrets import compare_digest
from typing import Literal

from fastapi import APIRouter, Query, Request
from pydantic import JsonValue
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from riskbyte_api.db.models import AuditEvent
from riskbyte_api.schemas.inference import (
    BatchInput,
    InvestigationPatch,
    PredictionResponse,
    TransactionInput,
    TransactionPage,
)
from riskbyte_ml.data.errors import DatasetError
from riskbyte_ml.explainability.schemas import UnifiedExplanation

router = APIRouter(prefix="/api/v1", tags=["RiskByte final API"])


def runtime(request):
    service = getattr(request.app.state, "runtime", None)
    if service is None:
        raise DatasetError("inference_unavailable", "Frozen inference service is not ready", 503)
    return service


def write_guard(request):
    key = request.app.state.settings.write_api_key
    if key is not None and not compare_digest(
        request.headers.get("X-Api-Key", ""), key.get_secret_value()
    ):
        raise DatasetError("write_access_denied", "Write API key required", 403)


@router.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput, request: Request):
    write_guard(request)
    return runtime(request).predict([transaction])[0]


@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(batch: BatchInput, request: Request):
    write_guard(request)
    return runtime(request).predict(batch.transactions)


@router.get("/transactions", response_model=TransactionPage)
def transactions(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    risk_level: Literal["LOW", "GUARDED", "ELEVATED", "HIGH", "CRITICAL"] | None = None,
    investigation_required: bool | None = None,
    sort: Literal["recent", "risk"] = "recent",
    search: str | None = Query(None, max_length=100),
    investigation_state: Literal["UNREVIEWED", "UNDER_REVIEW", "CLEARED", "ESCALATED"]
    | None = None,
):
    return runtime(request).listing(
        page, page_size, risk_level, investigation_required, sort, search, investigation_state
    )


@router.get("/transactions/{transaction_id}", response_model=dict[str, JsonValue])
def transaction_detail(transaction_id: str, request: Request):
    return runtime(request).detail(transaction_id)


@router.get("/transactions/{transaction_id}/explanation", response_model=UnifiedExplanation)
def explanation(transaction_id: str, request: Request):
    return runtime(request).explanation(transaction_id)


@router.patch("/transactions/{transaction_id}/investigation", response_model=dict[str, JsonValue])
def investigation(transaction_id: str, patch: InvestigationPatch, request: Request):
    write_guard(request)
    return runtime(request).investigate(transaction_id, patch.state)


@router.get("/dashboard/summary", response_model=dict[str, JsonValue])
def dashboard(request: Request):
    return runtime(request).dashboard()


@router.get("/audit", response_model=dict[str, JsonValue])
def audit_history(
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    transaction_id: str | None = Query(None, max_length=100),
):
    """Bounded operational events; only explicitly safe review metadata is exposed."""
    service = runtime(request)
    condition = [] if transaction_id is None else [AuditEvent.transaction_id == transaction_id]
    with Session(service.engine) as session:
        total = session.scalar(select(func.count()).select_from(AuditEvent).where(*condition))
        events = session.scalars(
            select(AuditEvent)
            .where(*condition)
            .order_by(AuditEvent.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "items": [
                {
                    "id": event.id,
                    "transaction_id": event.transaction_id,
                    "action": event.action,
                    "timestamp": (
                        event.created_at.astimezone(UTC)
                        if event.created_at.tzinfo
                        else event.created_at.replace(tzinfo=UTC)
                    ).isoformat(),
                    "metadata": {
                        k: v
                        for k, v in event.safe_metadata.items()
                        if k in {"previous", "from", "to", "state", "origin"}
                    },
                }
                for event in events
            ],
        }


@router.get("/evaluation/final", response_model=dict[str, JsonValue])
def final_evaluation(request: Request):
    return runtime(request).final_reports["final"]


@router.get("/evaluation/scenarios", response_model=dict[str, JsonValue])
def scenario_evaluation(request: Request):
    return runtime(request).final_reports["scenarios"]


@router.get("/evaluation/hard-negatives", response_model=dict[str, JsonValue])
def hard_negative_evaluation(request: Request):
    return runtime(request).final_reports["hard-negatives"]


@router.get("/system/model", response_model=dict[str, JsonValue])
def model_info(request: Request):
    return runtime(request).model_info()


@router.get("/system/health", response_model=dict[str, JsonValue])
def system_health(request: Request):
    service = getattr(request.app.state, "runtime", None)
    database_ready = False
    if service is not None:
        try:
            with service.engine.connect() as connection:
                database_ready = connection.scalar(text("SELECT 1")) == 1
        except Exception:
            database_ready = False
    return {
        "api_ready": True,
        "inference_ready": service is not None and database_ready,
        "supervised_loaded": service is not None,
        "anomaly_loaded": service is not None,
        "behaviour_loaded": service is not None,
        "abrf_loaded": service is not None,
        "shap_ready": service is not None,
        "database_ready": database_ready,
        "unavailable_reason": getattr(request.app.state, "runtime_error", None),
    }
