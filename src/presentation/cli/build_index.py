"""
Script CLI pour construire l'index FAISS LangChain.
Point d'entrée en ligne de commande pour la création de l'index vectoriel.
"""
from config.config import EMBEDDINGS_PATH, MISTRAL_API_KEY, EMBEDDING_MODEL, INDEX_PATH
from config.logger import logger
from langchain_mistralai import MistralAIEmbeddings
from src.rag.langchain_faiss_indexer import LangChainFaissIndexer

if __name__ == "__main__":
    logger.info("🚀 Démarrage de la construction de l'index FAISS...")
    
    # Initialisation du client d'embeddings
    logger.info("🤖 Initialisation du client Mistral...")
    embedding_client = MistralAIEmbeddings(
        api_key=MISTRAL_API_KEY,
        model=EMBEDDING_MODEL
    )

    # Initialisation de l'indexer
    logger.info("📚 Initialisation de l'indexer...")
    indexer = LangChainFaissIndexer(
        index_dir=INDEX_PATH,
        embedding_client=embedding_client
    )

    # Construction du vectorstore depuis les embeddings existants
    logger.info("🏗️ Construction du FAISS VectorStore...")
    indexer.build_from_embeddings_file(EMBEDDINGS_PATH)

    # Sauvegarde
    logger.info("💾 Sauvegarde de l'index...")
    indexer.save()

    logger.info("✅ Index LangChain créé et sauvegardé avec succès!")
