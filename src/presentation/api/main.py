"""
FastAPI application - Point d'entrée de l'API REST.
"""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from config.config import APP_TITLE
from src.presentation.api.routes import router

app = FastAPI(
    title=APP_TITLE,
    version='1.0.0',
    description='API RAG pour les événements culturels de Rennes Métropole'
)

app.include_router(router)

@app.get('/', tags=['API HEALTH'])
def root():
    """
    Redirige vers la documentation interactive Swagger UI.
    """
    return RedirectResponse(url="/docs")
