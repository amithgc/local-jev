"""HTTP layer: the Jev-compatible /v1 API, the project API behind the UI, and the UI itself."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .engine import Engine
from .models import ALIASES
from .questions import ContextOverflow, QuestionError
from .schema import ModelMetadataList, SystemOneRequest, SystemOneResponse

UI_DIR = Path(__file__).parent / "ui"
REQUEST_ID_HEADER = "x-typesafe-request-id"


def validation_error(loc, msg, kind="value_error", status=422):
    return JSONResponse(status_code=status, content={"detail": [{"loc": ["body", *loc], "msg": msg, "type": kind}]})


def create_app(engine: Engine | None = None, db_path: str | None = None, ui: bool = False) -> FastAPI:
    """The HTTP app. The Jev-compatible API is always on; ``ui`` adds the browser portal and the project API behind it."""
    engine = engine or Engine()
    api_key = os.environ.get("LOCAL_JEV_API_KEY")
    app = FastAPI(title="local-jev", version=__version__,
                  description="A local, Jev-compatible System One server built on small open models.")
    app.state.engine = engine

    @app.middleware("http")
    async def request_id_and_auth(request: Request, call_next):
        rid = uuid.uuid4().hex
        if api_key and request.url.path.startswith("/v1/"):
            if request.headers.get("authorization", "") != f"Bearer {api_key}":
                return JSONResponse(status_code=401, headers={REQUEST_ID_HEADER: rid},
                                    content={"detail": "Missing or invalid API key. Check the Authorization header."})
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        return response

    @app.post("/v1/systemone", response_model=SystemOneResponse, tags=["System One"])
    def system_one(body: SystemOneRequest, response: Response):
        """Evaluate one state against one or more questions. Same contract as Jev's endpoint.

        A state longer than the model can read is shortened, keeping its start and end; the response then
        carries the header ``x-local-jev-truncated: true``."""
        if engine.resolve(body.model) is None:
            return validation_error(["model"], f"Unknown model '{body.model}'. See GET /v1/models.", status=404)
        try:
            result = engine.system_one(body.state, body.questions, body.model)
            if result.pop("truncated", False):
                response.headers["x-local-jev-truncated"] = "true"
            return result
        except QuestionError as err:
            return validation_error(err.loc, err.msg)
        except ContextOverflow as err:
            return validation_error(["state"], str(err), "context_length_exceeded")
        except RuntimeError as err:                       # the model exists but cannot be loaded (missing weights or extra)
            return validation_error(["model"], str(err), "model_unavailable", status=503)

    @app.get("/v1/models", response_model=ModelMetadataList, tags=["System One"])
    def models():
        cards = [{"name": m.name, "description": m.description, "release_date": m.release_date}
                 for m in engine.models.values()]
        default = engine.models[engine.default_model]
        cards += [{"name": alias, "description": f"Alias of {default.name}.", "release_date": default.release_date}
                  for alias in ALIASES]
        return {"models": cards}

    @app.get("/healthz", include_in_schema=False)
    def health():
        return {"ok": True, "version": __version__, "default_model": engine.default_model}

    if not ui:
        @app.get("/", include_in_schema=False)
        def root():
            return {"name": "local-jev", "version": __version__, "api": "/v1/systemone", "models": "/v1/models",
                    "docs": "/docs", "ui": "off; start the server with --ui to enable the browser portal"}
        return app

    from .projects import attach as attach_projects
    attach_projects(app, engine, db_path)
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(UI_DIR / "index.html")

    return app
