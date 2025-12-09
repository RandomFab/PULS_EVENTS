from fastapi import APIRouter
from src.app.services.rag_service import retriever

router = APIRouter()

@router.get('/health_router')
def health_router():
    """
    Retourne le statut de santé de l'API via router
    """
    return {'status':'Router ok'}

@router.get('/index_info')
def get_index_info():
    """
    Retourne les informations de l'indexer
    """
    return retriever.index_info()

@router.post('/ask')
def ask(query):
    response = retriever.answer_query(query=query)

    return response