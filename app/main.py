"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings

app = FastAPI(title=settings.app_name)
app.include_router(router)