from src.rag.langchain_faiss_indexer import LangChainFaissIndexer
from src.rag.retriever import SimpleRetriever
from src.rag.embedder import Embedder
from config.config import MISTRAL_API_KEY,INDEX_PATH,MODEL_NAME,EMBEDDING_MODEL
from config.logger import logger

retriever = SimpleRetriever(INDEX_PATH,api_key=MISTRAL_API_KEY,embedding_model=EMBEDDING_MODEL,model_name=MODEL_NAME)

try: 
    logger.info("🔄 Chargement de l'indexer")
    retriever.indexer.load()
    logger.info("✅ L'indexer a été chargé avec succès")
except Exception as e:
    logger.error(f"❌ Echec du chargement de l'indexer → {e}")