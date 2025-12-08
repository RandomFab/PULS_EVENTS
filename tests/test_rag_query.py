from src.rag.faiss_indexer import FaissIndexer
from src.rag.embedder import Embedder

from config.config import INDEX_PATH, METADATA_PATH,MISTRAL_API_KEY,EMBEDDING_MODEL
from config.logger import logger

embedder = Embedder(api_key=MISTRAL_API_KEY,model=EMBEDDING_MODEL)
indexer = FaissIndexer(INDEX_PATH,METADATA_PATH)

query = 'festival jazz cesson'
_,query_vector = embedder.chunk_and_embed(query)

indexer.load_index()

results = indexer.search(query_vector=query_vector,k=5)

for i, result in enumerate(results[0]):
    logger.info(f"[Top {i+1}] Meilleur résultat pour la question : {query} \n contexte : {result['chunk']} \n Ville : {result['location_address']} \n ============================================ ")