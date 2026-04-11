"""FastAPI application for the Clinical Note Scribe environment."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server.routes import router

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(
    title="Clinical Note Scribe – OpenEnv",
    description="OpenEnv-compliant environment for evaluating AI agents on clinical SOAP-note generation.",
    version="1.0.0",
)
app.include_router(router)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
