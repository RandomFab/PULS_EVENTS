from fastapi import APIRouter
from src.app.services.rag_service import retriever
from fastapi import HTTPException, status
from config.config import EMBEDDINGS_PATH

router = APIRouter()

@router.get('/health_router',summary="S'assurer que l'API fonctionne")
def health_router():
    """
    Retourne le statut de santé de l'API via router
    """
    return {'status':'Router ok'}

@router.get('/index_info',summary="Consulter les info de l'index Faiss")
def get_index_info():
    """
    Retourne les informations de l'indexer
    """
    return retriever.indexer.index_info()

@router.post('/ask',summary="Poser une questions")
def ask(query:str = 'Quels sont les évenements rock à rennes ?'):
    """
    Permet de poser une question concernant les évenement musicaux de Rennes.
    Renvoie une réponse généré par LLM sur la base des évenements de Rennes
    
    :param query: Description
    :type query: str
    """
    response = retriever.answer_query(query=query)

    return {
        "status": "success",
        "data": {
            "message": "Voici les événements trouvés :",
            "formatted_response": response  # Texte brut avec \n
        }
    }

@router.post('/rebuild',summary="Re créer l'index")
def rebuild():
    """
    Permet de re créer ou re généré la base index Faiss
    peut être précédé d'une mise à jour de la base de données OpenAgenda
    """
    try:
        retriever.indexer.build_from_embeddings_file(EMBEDDINGS_PATH)
        infos = get_index_info()
        return {'status': 'success', 'message' : f"✅ Le rebuild a été effectué avec succès. \n Nouvelles informations de l'index : {infos} "}
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="❌ Le fichier n'a pas été trouvé"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur est survenue : {str(e)}"
            )
    
@router.post('/search_raw')
def search_raw(query:str = "concert jazz Pacé"):
    """
    Permet de récupérer les K churns les plus proche de la requête.
    Non récupérons donc des churns bruts, non retravaillé par un LLM
    
    :param query: Description
    :type query: str
    """
    try:

        results = retriever.indexer.search(query=query,k=5,score_threshold=None)
        churns = [{'event_id' : docs.metadata.get('event_id'),'title':docs.metadata.get('title'),'location_address':docs.metadata.get('location_address')} for docs, score in results]
        return {'status':'success', 'message':f"Voici les {len(churns)} résultats les plus proche : {churns}"}
    except Exception as e :
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur est survenue : {str(e)}"
        )
