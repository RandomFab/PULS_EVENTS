"""
Tests pour l'API.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


class TestAPIStartup:
    """Tests du démarrage de l'API"""
    
    def test_api_main_is_importable(self):
        """Vérifie que le module main est importable"""
        from src.presentation.api import main
        assert main is not None
    
    def test_app_is_fastapi_instance(self):
        """Vérifie que l'app est bien une instance FastAPI"""
        from src.presentation.api.main import app
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)


class TestAPIRoutes:
    """Tests basiques des routes"""
    
    def test_routes_module_is_importable(self):
        """Vérifie que le module routes est importable"""
        from src.presentation.api import routes
        assert routes is not None
