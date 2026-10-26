"""Consistent, sanitized HTTP error envelopes."""

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from riskbyte_api.schemas.errors import ErrorDetail, ErrorResponse, ValidationIssue
from riskbyte_ml.data.errors import DatasetError

logger = logging.getLogger(__name__)


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    issues: list[ValidationIssue] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unavailable")
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, request_id=request_id, issues=issues)
    )
    response_headers = dict(headers or {})
    response_headers["X-Request-ID"] = request_id
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_none=True),
        headers=response_headers,
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    try:
        message = HTTPStatus(exc.status_code).phrase
    except ValueError:
        message = "Request failed"
    return error_response(
        request, exc.status_code, f"http_{exc.status_code}", message, headers=exc.headers
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    issues = [
        ValidationIssue(location=list(issue["loc"]), code=issue["type"]) for issue in exc.errors()
    ]
    return error_response(request, 422, "validation_error", "Request validation failed", issues)


async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled request error",
        extra={
            "request_id": getattr(request.state, "request_id", "unavailable"),
            "error_type": type(exc).__name__,
        },
    )
    return error_response(request, 500, "internal_error", "An internal error occurred")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)
    app.add_exception_handler(DatasetError, dataset_exception_handler)


async def dataset_exception_handler(request: Request, exc: DatasetError) -> JSONResponse:
    return error_response(request, exc.http_status, exc.code, exc.message)
