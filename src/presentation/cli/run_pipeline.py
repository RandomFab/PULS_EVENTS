"""
Script CLI principal - Pipeline complet de traitement.
Orchestre toutes les étapes : récupération des données, génération des embeddings, construction de l'index.
"""
import argparse
from config.config import (
    OPENAGENDA_API_KEY, RAW_DATA, PROCESSED_DATA,
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDINGS_PATH, INDEX_PATH
)
from config.logger import logger
from src.infrastructure.openagenda_client import OpenAgendaClient
from src.domain.services.event_service import EventService
from src.domain.services.embedding_service import EmbeddingService
from src.rag.embedder import Embedder
from langchain_mistralai import MistralAIEmbeddings
from src.rag.langchain_faiss_indexer import LangChainFaissIndexer


def run_full_pipeline(start_date: str = '01/01/25', end_date: str = '01/01/27'):
    """
    Exécute le pipeline complet de traitement des événements.
    
    Args:
        start_date: Date de début au format DD/MM/YY
        end_date: Date de fin au format DD/MM/YY
    """
    logger.info("=" * 80)
    logger.info("🚀 DÉMARRAGE DU PIPELINE COMPLET")
    logger.info("=" * 80)
    
    try:
        # ========== ÉTAPE 1 : RÉCUPÉRATION DES ÉVÉNEMENTS ==========
        logger.info("\n📡 ÉTAPE 1/3 : Récupération des événements OpenAgenda")
        logger.info("-" * 80)
        
        openagenda_client = OpenAgendaClient(api_key=OPENAGENDA_API_KEY)
        event_service = EventService(
            openagenda_client=openagenda_client,
            raw_data_path=RAW_DATA,
            processed_data_path=PROCESSED_DATA
        )
        
        df = event_service.fetch_and_process_events(
            start_date=start_date,
            end_date=end_date
        )
        logger.info(f"✅ Étape 1 terminée : {len(df)} événements traités")
        
        # ========== ÉTAPE 2 : GÉNÉRATION DES EMBEDDINGS ==========
        logger.info("\n🧠 ÉTAPE 2/3 : Génération des embeddings")
        logger.info("-" * 80)
        
        embedder = Embedder(api_key=MISTRAL_API_KEY, model=EMBEDDING_MODEL)
        embedding_service = EmbeddingService(
            embedder=embedder,
            processed_data_path=PROCESSED_DATA,
            embeddings_path=EMBEDDINGS_PATH,
            batch_size=10
        )
        
        embedding_service.build_embeddings()
        logger.info("✅ Étape 2 terminée : Embeddings générés et sauvegardés")
        
        # ========== ÉTAPE 3 : CONSTRUCTION DE L'INDEX FAISS ==========
        logger.info("\n📚 ÉTAPE 3/3 : Construction de l'index FAISS")
        logger.info("-" * 80)
        
        embedding_client = MistralAIEmbeddings(
            api_key=MISTRAL_API_KEY,
            model=EMBEDDING_MODEL
        )
        
        indexer = LangChainFaissIndexer(
            index_dir=INDEX_PATH,
            embedding_client=embedding_client
        )
        
        indexer.build_from_embeddings_file(EMBEDDINGS_PATH)
        indexer.save()
        
        infos = indexer.index_info()
        logger.info(f"✅ Étape 3 terminée : Index créé avec {infos}")
        
        # ========== RÉSUMÉ ==========
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PIPELINE TERMINÉ AVEC SUCCÈS !")
        logger.info("=" * 80)
        logger.info(f"📊 Résumé :")
        logger.info(f"  - Événements traités : {len(df)}")
        logger.info(f"  - Fichiers générés :")
        logger.info(f"    • {RAW_DATA}")
        logger.info(f"    • {PROCESSED_DATA}")
        logger.info(f"    • {EMBEDDINGS_PATH}")
        logger.info(f"    • {INDEX_PATH}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"\n❌ ERREUR CRITIQUE : {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pipeline complet de traitement des événements RAG'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='01/01/25',
        help='Date de début (format: DD/MM/YY). Défaut: 01/01/25'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='01/01/27',
        help='Date de fin (format: DD/MM/YY). Défaut: 01/01/27'
    )
    
    args = parser.parse_args()
    
    run_full_pipeline(
        start_date=args.start_date,
        end_date=args.end_date
    )
