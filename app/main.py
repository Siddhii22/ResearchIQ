"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router
from app.api.research import router as research_router
from app.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(router)
app.include_router(research_router)