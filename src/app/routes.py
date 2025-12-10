from fastapi import APIRouter
from src.app.services.rag_service import retriever
from src.app.services.data_service import update_events
from src.app.services.embed_service import build_embeddings
from fastapi import HTTPException, status
from config.config import EMBEDDINGS_PATH
from pydantic import BaseModel, Field,field_validator
import datetime

class Ask(BaseModel):
    query: str = Field('Quels sont les évenements rock à rennes ?', max_length=200)
class SearchRaw(BaseModel):
    query: str = Field('concert jazz Pacé', max_length=100)
class UpdateDate(BaseModel):
    start_date:str = Field('01/01/25',pattern=r"^\d{2}/\d{2}/\d{2}$")
    end_date:str = Field('31/12/25',pattern=r"^\d{2}/\d{2}/\d{2}$")

    @field_validator("start_date","end_date")
    def check_valid_date(cls, v):
        try:
            datetime.datetime.strptime(v, "%d/%m/%y")
        except ValueError:
            raise ValueError("La date doit être valide et au format JJ/MM/AA.")
        return v

router = APIRouter()

@router.get('/health_router',summary="S'assurer que l'API fonctionne",tags=['API HEALTH'])
def health_router():
    """
    Retourne le statut de santé de l'API via router
    """
    return {'status':'Router ok'}

@router.get('/index_info',summary="Consulter les info de l'index Faiss",tags=['GET INFOS'])
def get_index_info():
    """
    Retourne les informations de l'indexer
    """
    return retriever.indexer.index_info()

@router.post('/ask',summary="Poser une questions",tags=['USE RAG'])
def ask(payload:Ask):
    """
    Permet de poser une question concernant les évenement musicaux de Rennes.
    Renvoie une réponse généré par LLM sur la base des évenements de Rennes
    """
    response = retriever.answer_query(query=payload.query)

    return {
        "status": "success",
        "data": {
            "message": "Voici les événements trouvés :",
            "formatted_response": response  # Texte brut avec \n
        }
    }

@router.post('/rebuild',summary="Re créer l'index",tags=['USE RAG'])
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
    
@router.post('/search_raw',tags=['USE RAG'])
def search_raw(payload:SearchRaw):
    """
    Permet de récupérer les K churns les plus proche de la requête.
    Non récupérons donc des churns bruts, non retravaillé par un LLM
    """
    try:

        results = retriever.indexer.search(query=payload.query,k=5,score_threshold=None)
        churns = [{'event_id' : docs.metadata.get('event_id'),'title':docs.metadata.get('title'),'location_address':docs.metadata.get('location_address')} for docs, score in results]
        return {'status':'success', 'message':f"Voici les {len(churns)} résultats les plus proche : {churns}"}
    except Exception as e :
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur est survenue : {str(e)}"
        )

@router.post('/update_datas',tags=['USE RAG'])
def update_datas(dates:UpdateDate):
    """Met à jour les événements entre deux dates, reconstruit les embeddings et rebuild l'index.
    Retourne les informations de l'index mis à jour ou lève une HTTPException en cas d'erreur.
    """
    try:
        update_events(start_date=dates.start_date,end_date=dates.end_date)
        build_embeddings(retriever.embedder)
        rebuild()
        infos = get_index_info()
        return {'status':'success','message' : f"✅ La mise a jour des données et le rebuild ont été effectués avec succès. \n Nouvelles informations de l'index : {infos} "}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur est survenue : {str(e)}"
        )