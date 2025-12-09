from src.rag.retriever import SimpleRetriever
from config.config import MISTRAL_API_KEY, EMBEDDING_MODEL, MODEL_NAME,INDEX_PATH

retrieverQA = SimpleRetriever(index_dir=INDEX_PATH,api_key=MISTRAL_API_KEY,embedding_model=EMBEDDING_MODEL,model_name=MODEL_NAME)

print("\n" + "="*60)
print("📌 EXEMPLE 1 : Réponse simple")
print("="*60)
retrieverQA.answer_query('y a t il des évenements jazz à Pacé ?')