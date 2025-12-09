from fastapi import APIRouter

router = APIRouter()

@router.get('/health_router')
def health_router():
    """
    Retourne le statut de santé de l'API via router
    """
    return {'status':'Router ok'}