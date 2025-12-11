"""
Script CLI pour générer les embeddings des événements.
Point d'entrée en ligne de commande pour la création des vecteurs d'embeddings.
"""
from config.config import MISTRAL_API_KEY, EMBEDDING_MODEL, PROCESSED_DATA, EMBEDDINGS_PATH
from config.logger import logger
from src.rag.embedder import Embedder
from src.domain.services.embedding_service import EmbeddingService

if __name__ == "__main__":
    logger.info("🚀 Démarrage de la génération des embeddings...")
    
    # Initialisation de l'embedder (RAG layer)
    embedder = Embedder(api_key=MISTRAL_API_KEY, model=EMBEDDING_MODEL)
    
    # Initialisation du service
    embedding_service = EmbeddingService(
        embedder=embedder,
        processed_data_path=PROCESSED_DATA,
        embeddings_path=EMBEDDINGS_PATH,
        batch_size=10
    )
    
    # Exécution
    embedding_service.build_embeddings()
    
    logger.info("✅ Génération des embeddings terminée !")
