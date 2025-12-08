from src.rag.faiss_indexer import FaissIndexer
from config.config import METADATA_PATH, INDEX_PATH,EMBEDDINGS_PATH

faissindexer = FaissIndexer(INDEX_PATH,METADATA_PATH)

faissindexer.index_embeddings(EMBEDDINGS_PATH)
faissindexer.save_index()