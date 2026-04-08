"""FastAPI application for the Clinical Note Scribe environment.

Run locally::

    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

Or via Docker (see ``Dockerfile`` in project root).
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI

from server.routes import router

# ---------------------------------------------------------------------------
# Configure root logging → structured JSON to stdout
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Silence noisy uvicorn access logs so our structured events stay clean
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Clinical Note Scribe – OpenEnv",
    description=(
        "An OpenEnv-compliant environment for evaluating AI agents on "
        "clinical SOAP-note generation from doctor–patient transcripts."
    ),
    version="0.1.0",
)

from fastapi.responses import RedirectResponse

# Mount all routes at root (/)
app.include_router(router)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the FastAPI interactive documentation."""
    return RedirectResponse(url="/docs")

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
