from src.rag.langchain_faiss_indexer import LangChainFaissIndexer
from src.rag.embedder import Embedder
from config.config import MISTRAL_API_KEY,INDEX_PATH
from config.logger import logger

embbeding_client = Embedder(api_key=MISTRAL_API_KEY)
indexer = LangChainFaissIndexer(index_dir=INDEX_PATH,embedding_client=embbeding_client)

try: 
    logger.info("🔄 Chargement de l'indexer")
    indexer.load()
    logger.info("✅ L'indexer a été chargé avec succès")
except Exception as e:
    logger.error(f"❌ Echec du chargement de l'indexer → {e}")