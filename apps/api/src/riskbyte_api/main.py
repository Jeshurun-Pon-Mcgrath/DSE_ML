"""ASGI application factory; run with uvicorn riskbyte_api.main:create_app --factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from riskbyte_api import __version__
from riskbyte_api.core.body_limit import BodyLimitMiddleware
from riskbyte_api.core.config import Settings
from riskbyte_api.core.exceptions import register_exception_handlers
from riskbyte_api.core.logging import configure_logging
from riskbyte_api.core.middleware import RequestIdMiddleware
from riskbyte_api.routers.abrf import router as abrf_router
from riskbyte_api.routers.dataset import router as dataset_router
from riskbyte_api.routers.explainability import router as explainability_router
from riskbyte_api.routers.health import router as health_router
from riskbyte_api.routers.inference import router as inference_router
from riskbyte_api.routers.models import router as model_router
from riskbyte_api.routers.phase5 import router as phase5_router
from riskbyte_api.routers.statistics import router as statistics_router
from riskbyte_api.schemas.errors import ErrorResponse
from riskbyte_api.services.abrf import ABRFService
from riskbyte_api.services.dataset import DatasetService
from riskbyte_api.services.explainability import ExplainabilityService
from riskbyte_api.services.models import ModelService
from riskbyte_api.services.phase5 import Phase5Service
from riskbyte_api.services.statistics import StatisticsService


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else Settings()
    configure_logging(config.log_level)

    @asynccontextmanager
    async def lifespan(application):
        if config.inference_enabled and config.dataset_config_path:
            from riskbyte_api.services.runtime import RuntimeService

            try:
                application.state.runtime = RuntimeService(config)
                application.state.database_engine = application.state.runtime.engine
            except Exception:
                application.state.runtime_error = "frozen_runtime_startup_failed"
        yield
        if application.state.runtime is not None:
            application.state.runtime.engine.dispose()

    app = FastAPI(
        title="RiskByte API",
        version=__version__,
        description="Frozen synthetic RiskByte inference, final evaluation and analyst workflow.",
        debug=False,
        lifespan=lifespan,
        docs_url="/docs" if config.environment != "production" else None,
        redoc_url=None,
        openapi_url="/openapi.json" if config.environment != "production" else None,
        responses={
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    app.state.settings = config
    app.state.runtime = None
    app.state.runtime_error = None
    app.state.dataset_service = DatasetService(config)
    app.state.statistics_service = StatisticsService(config)
    app.state.model_service = ModelService(config)
    app.state.phase5_service = Phase5Service(config)
    app.state.abrf_service = ABRFService(config)
    app.state.explainability_service = ExplainabilityService(config)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(dataset_router)
    app.include_router(statistics_router)
    app.include_router(model_router)
    app.include_router(phase5_router)
    app.include_router(abrf_router)
    app.include_router(explainability_router)
    app.include_router(inference_router)
    app.add_middleware(BodyLimitMiddleware, max_bytes=config.max_request_bytes)
    app.add_middleware(RequestIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Content-Type", "X-Api-Key"],
        expose_headers=["X-Request-ID"],
    )
    return app
