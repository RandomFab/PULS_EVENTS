"""
Routes FastAPI - Endpoints de l'API REST.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import datetime

from config.config import (
    EMBEDDINGS_PATH, SEARCH_K, MISTRAL_API_KEY, EMBEDDING_MODEL,
    MODEL_NAME, INDEX_PATH, PROCESSED_DATA, RAW_DATA, OPENAGENDA_API_KEY
)
from config.logger import logger

# Import des services du domaine
from src.domain.services.event_service import EventService
from src.domain.services.embedding_service import EmbeddingService
from src.domain.services.rag_service import RagService

# Import des composants RAG et infrastructure
from src.infrastructure.openagenda_client import OpenAgendaClient
from src.rag.embedder import Embedder
from src.rag.retriever import SimpleRetriever

# Initialisation de l'infrastructure
openagenda_client = OpenAgendaClient(api_key=OPENAGENDA_API_KEY)
embedder = Embedder(api_key=MISTRAL_API_KEY, model=EMBEDDING_MODEL)
retriever = SimpleRetriever(
    index_dir=INDEX_PATH,
    api_key=MISTRAL_API_KEY,
    embedding_model=EMBEDDING_MODEL,
    model_name=MODEL_NAME
)

# Initialisation des services du domaine
event_service = EventService(
    openagenda_client=openagenda_client,
    raw_data_path=RAW_DATA,
    processed_data_path=PROCESSED_DATA
)

embedding_service = EmbeddingService(
    embedder=embedder,
    processed_data_path=PROCESSED_DATA,
    embeddings_path=EMBEDDINGS_PATH,
    batch_size=10
)

rag_service = RagService(retriever=retriever)

# Modèles Pydantic pour validation
class Ask(BaseModel):
    query: str = Field('Quels sont les évenements rock à rennes ?', max_length=200)

class SearchRaw(BaseModel):
    query: str = Field('concert jazz Pacé', max_length=100)

class UpdateDate(BaseModel):
    start_date: str = Field('01/01/25', pattern=r"^\d{2}/\d{2}/\d{2}$")
    end_date: str = Field('31/12/25', pattern=r"^\d{2}/\d{2}/\d{2}$")

    @field_validator("start_date", "end_date")
    def check_valid_date(cls, v):
        try:
            datetime.datetime.strptime(v, "%d/%m/%y")
        except ValueError:
            raise ValueError("La date doit être valide et au format JJ/MM/AA.")
        return v

router = APIRouter()

# ========== HEALTH & INFO ==========

@router.get('/health_router', summary="Vérifier l'état de l'API", tags=['API HEALTH'])
def health_router():
    """
    Retourne le statut de santé de l'API via router.
    """
    return {'status': 'Router ok'}

@router.get('/index_info', summary="Consulter les infos de l'index FAISS", tags=['GET INFOS'])
def get_index_info():
    """
    Retourne les informations de l'indexer (nombre de documents, dimension, etc.).
    """
    try:
        infos = retriever.indexer.index_info()
        if not infos:
            logger.error("Index FAISS introuvable ou vide dans /index_info")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Index FAISS introuvable ou vide."
            )
        return infos
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Erreur inattendue dans /index_info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des infos de l'index : {str(e)}"
        )

# ========== RAG OPERATIONS ==========

@router.post('/ask', summary="Poser une question", tags=['USE RAG'])
def ask(payload: Ask):
    """
    Permet de poser une question concernant les événements musicaux de Rennes.
    Renvoie une réponse générée par le LLM sur la base des événements de Rennes.
    """
    try:
        response = rag_service.search_and_answer(query=payload.query)
        if not response or not isinstance(response,str):
            logger.error("Réponse LLM vide")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="La génération de la réponse a échoué. Vérifiez l'API Mistral ou les données."
            )
        
        return {
            "status": "success",
            "data": {
                "message": "Voici les événements trouvés :",
                "formatted_response": response
            }
        }
    except HTTPException as e:
        # Erreur métier ou validation déjà formatée
        raise e
    except ValueError as e:
        logger.error(f"Erreur de validation dans /ask: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur de validation : {str(e)}"
        )
    except ConnectionError as e:
        logger.error(f"Erreur de connexion à l'API Mistral dans /ask: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Impossible de joindre l'API Mistral. Réessayez plus tard."
        )
    except Exception as e:
        logger.error(f"❌ Erreur inattendue dans /ask: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur inattendue lors de la génération de la réponse : {str(e)}"
        )

@router.post('/search_raw', summary="Recherche brute dans l'index", tags=['USE RAG'])
def search_raw(payload: SearchRaw):
    """
    Permet de récupérer les K chunks les plus proches de la requête.
    Retourne des chunks bruts, non retravaillés par un LLM.
    """
    try:
        if not payload.query or len(payload.query.strip()) < 3:
            logger.warning("Requête trop courte ou vide dans /search_raw")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="La requête doit contenir au moins 3 caractères."
            )
        results = retriever.indexer.search(
            query=payload.query,
            k=SEARCH_K,
            score_threshold=None
        )
        chunks = [
            {
                'event_id': doc.metadata.get('event_id'),
                'title': doc.metadata.get('title'),
                'location_address': doc.metadata.get('location_address')
            }
            for doc, score in results
        ]
        if not chunks:
            logger.warning("Aucun résultat pertinent trouvé dans /search_raw")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Aucun résultat pertinent trouvé."
            )
        return {
            'status': 'success',
            'message': f"Voici les {len(chunks)} résultats les plus proches : {chunks}"
        }
    except HTTPException as e:
        raise e
    except ValueError as e:
        logger.error(f"Erreur de validation dans /search_raw: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur de validation : {str(e)}"
        )
    except ConnectionError as e:
        logger.error(f"Erreur de connexion à l'index FAISS dans /search_raw: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Impossible de joindre l'index FAISS. Réessayez plus tard."
        )
    except Exception as e:
        logger.error(f"❌ Erreur inattendue dans /search_raw: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur inattendue lors de la recherche : {str(e)}"
        )

@router.post('/rebuild', summary="Recréer l'index FAISS", tags=['USE RAG'])
def rebuild():
    """
    Permet de recréer ou régénérer la base index FAISS.
    Peut être précédé d'une mise à jour de la base de données OpenAgenda.
    """
    try:
        retriever.indexer.build_from_embeddings_file(EMBEDDINGS_PATH)
        retriever.indexer.save()
        infos = retriever.indexer.index_info()
        if not infos:
            logger.error("Index FAISS non créé ou vide dans /rebuild")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="L'index FAISS n'a pas pu être créé ou est vide."
            )
        return {
            'status': 'success',
            'message': f"✅ Le rebuild a été effectué avec succès.\nNouvelles informations de l'index : {infos}"
        }
    except FileNotFoundError:
        logger.error("Fichier embeddings.json introuvable dans /rebuild")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="❌ Le fichier embeddings.json n'a pas été trouvé"
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Erreur inattendue dans /rebuild: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur inattendue lors du rebuild : {str(e)}"
        )

# ========== DATA MANAGEMENT ==========

@router.post('/update_datas', summary="Mettre à jour les données", tags=['USE RAG'])
def update_datas(dates: UpdateDate):
    """
    Met à jour les événements entre deux dates, reconstruit les embeddings et rebuild l'index.
    Retourne les informations de l'index mis à jour ou lève une HTTPException en cas d'erreur.
    """
    try:
        logger.info(f"🔄 Mise à jour des données du {dates.start_date} au {dates.end_date}")
        # Validation métier sur les dates
        if not dates.start_date or not dates.end_date:
            logger.warning("Dates manquantes dans /update_datas")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Les dates de début et de fin sont obligatoires."
            )
        # 1. Récupération des événements
        event_service.fetch_and_process_events(
            start_date=dates.start_date,
            end_date=dates.end_date
        )
        # 2. Génération des embeddings
        embedding_service.build_embeddings()
        # 3. Reconstruction de l'index
        retriever.indexer.build_from_embeddings_file(EMBEDDINGS_PATH)
        retriever.indexer.save()
        infos = retriever.indexer.index_info()
        if not infos:
            logger.error("Index FAISS non créé ou vide dans /update_datas")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="L'index FAISS n'a pas pu être créé ou est vide."
            )
        return {
            'status': 'success',
            'message': f"✅ La mise à jour des données et le rebuild ont été effectués avec succès.\nNouvelles informations de l'index : {infos}"
        }
    except HTTPException as e:
        raise e
    except ValueError as e:
        logger.error(f"Erreur de validation dans /update_datas: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur de validation : {str(e)}"
        )
    except FileNotFoundError as e:
        logger.error(f"Fichier manquant dans /update_datas: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fichier manquant : {str(e)}"
        )
    except ConnectionError as e:
        logger.error(f"Erreur de connexion dans /update_datas: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Impossible de joindre une ressource externe. Réessayez plus tard."
        )
    except Exception as e:
        logger.error(f"❌ Erreur inattendue dans /update_datas: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur inattendue lors de la mise à jour des données : {str(e)}"
        )

# ========== EVALUATION ==========

@router.get('/evaluate_rag', summary="Évaluer les performances du système RAG", tags=['EVALUATION'])
def evaluate_rag():
    """
    Lance l'évaluation du système RAG en utilisant Ragas.
    Retourne les métriques de performance ou un message d'erreur détaillé.
    """
    try:
        from src.domain.services.evaluation_service import evaluate_current_rag
        result = evaluate_current_rag()
        if result.get("status") == "error":
            # Déterminer le code HTTP approprié selon le type d'erreur
            if result.get("error_type") == "FileNotFoundError":
                status_code = status.HTTP_404_NOT_FOUND
            elif result.get("error_type") == "ValueError":
                status_code = status.HTTP_400_BAD_REQUEST
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            logger.error(f"Erreur métier dans /evaluate_rag: {result.get('message')}")
            raise HTTPException(
                status_code=status_code,
                detail=result.get("message", "Erreur inconnue")
            )
        if not result:
            logger.error("Aucun résultat d'évaluation retourné dans /evaluate_rag")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Aucun résultat d'évaluation retourné."
            )
        return result
    except HTTPException as e:
        raise e
    except FileNotFoundError as e:
        logger.error(f"Fichier testset manquant dans /evaluate_rag: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Fichier testset manquant : {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Erreur de validation dans /evaluate_rag: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erreur de validation : {str(e)}"
        )
    except ConnectionError as e:
        logger.error(f"Erreur de connexion dans /evaluate_rag: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Impossible de joindre une ressource externe. Réessayez plus tard."
        )
    except Exception as e:
        logger.error(f"❌ Erreur inattendue dans /evaluate_rag: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur inattendue lors de l'évaluation : {str(e)}"
        )
