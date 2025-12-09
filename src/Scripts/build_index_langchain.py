from src.rag.langchain_faiss_indexer import LangChainFaissIndexer
from langchain_mistralai import MistralAIEmbeddings
from config.config import EMBEDDINGS_PATH, MISTRAL_API_KEY, EMBEDDING_MODEL,INDEX_PATH
from config.logger import logger

# Initialisation du client d'embeddings
logger.info("🚀 Initialisation du client Mistral...")
embedding_client = MistralAIEmbeddings(
    api_key=MISTRAL_API_KEY,
    model=EMBEDDING_MODEL
)

# Initialisation de l'indexer
index_dir = INDEX_PATH  # Ou utilisez votre config
indexer = LangChainFaissIndexer(
    index_dir=index_dir,
    embedding_client=embedding_client
)

# Construction du vectorstore depuis les embeddings existants
logger.info("🏗️ Construction du FAISS VectorStore...")
indexer.build_from_embeddings_file(EMBEDDINGS_PATH)

# Sauvegarde
indexer.save()

logger.info("✅ Index LangChain créé et sauvegardé avec succès!")
