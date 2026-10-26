from fastapi import APIRouter

from riskbyte_api import __version__
from riskbyte_api.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Process liveness only; does not assert database or model readiness."""
    return HealthResponse(version=__version__)
