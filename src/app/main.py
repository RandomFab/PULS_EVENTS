from fastapi import FastAPI
from config.config import APP_TITLE
from src.app.routes import router

app = FastAPI(
    title=APP_TITLE,
    version='1.0.0'
)
app.include_router(router)

@app.get('/',tags=['API HEALTH'])
def root():
    """
    Message renvoyé sur le chemin root
    """
    return {f"Bonjour et bienvenue sur l'API {APP_TITLE}"}
