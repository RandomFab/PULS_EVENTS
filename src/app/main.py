from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from config.config import APP_TITLE
from src.app.routes import router

app = FastAPI(
    title=APP_TITLE,
    version='1.0.0'
)
app.include_router(router)

@app.get('/', tags=['API HEALTH'])
def root():
    """
    Redirige vers la documentation interactive
    """
    return RedirectResponse(url="/docs")
