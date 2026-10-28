"""GrowthPilot web package — ``uvicorn src.web:app`` keeps working."""

from src.web.app import app, create_app

__all__ = ["app", "create_app"]
