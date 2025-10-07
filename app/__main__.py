from __future__ import annotations

import os

import uvicorn

from .main import app


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=bool(os.getenv("UVICORN_RELOAD", "")),
    )
