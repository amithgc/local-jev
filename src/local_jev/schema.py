"""Jev's wire format, mirrored from the public OpenAPI schema the official SDK is generated from."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

Json = str | dict[str, Any] | list[Any]


class SystemOneRequest(BaseModel):
    state: Json = Field(..., description="The content all questions in this request refer to.")
    model: str = Field("jev-latest", description="Name or alias of the model to use.")
    # Questions are validated by the compiler so error locations point inside each question.
    questions: dict[str, dict[str, Any]] = Field(..., min_length=1)


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, dict[str, Any]]
    usage: Usage


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]
