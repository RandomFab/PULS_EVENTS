"""
Exemple d'utilisation du LangChainFaissIndexer pour la recherche
"""
from src.rag.langchain_faiss_indexer import LangChainFaissIndexer
from langchain_mistralai import MistralAIEmbeddings
from config.config import MISTRAL_API_KEY, EMBEDDING_MODEL, INDEX_PATH
from config.logger import logger

# Initialisation
embedding_client = MistralAIEmbeddings(
    api_key=MISTRAL_API_KEY,
    model=EMBEDDING_MODEL
)

indexer = LangChainFaissIndexer(
    index_dir=INDEX_PATH,
    embedding_client=embedding_client
)

# Chargement de l'index
logger.info("🔄 Chargement de l'index...")
indexer.load()

# 🔍 Exemple 1 : Recherche simple
print("\n" + "="*60)
print("📌 EXEMPLE 1 : Recherche simple")
print("="*60)
results = indexer.search("concert de musique rock", k=3)

for i, (doc, score) in enumerate(results, 1):
    print(f"\n🎯 Résultat {i} (score: {score:.4f})")
    print(f"   Titre: {doc.metadata['title']}")
    print(f"   Lieu: {doc.metadata['location_name']}")
    print(f"   Date: {doc.metadata['start_date']}")
    print(f"   Extrait: {doc.page_content[:150]}...")

# 🔍 Exemple 2 : Recherche avec seuil de score
print("\n" + "="*60)
print("📌 EXEMPLE 2 : Recherche avec seuil")
print("="*60)
results = indexer.search(
    "festival rap",
    k=5,
    score_threshold=0.8  # Seulement les résultats très pertinents
)

print(f"\n✅ {len(results)} résultats avec score <= 0.8")
for doc, score in results:
    print(f"   • {doc.metadata['title']} (score: {score:.4f})")

# 🔍 Exemple 3 : Recherche avec filtres métadonnées
print("\n" + "="*60)
print("📌 EXEMPLE 3 : Recherche avec filtres")
print("="*60)
try:
    results = indexer.filter_by_metadata(
        query="événement culturel",
        k=5,
        filter_dict={'location_address': 'Pacé'}  # Seulement à Pacé
    )
    
    print(f"\n✅ {len(results)} résultats à Pacé")
    for doc in results:
        print(f"   • {doc.metadata['title']}")
except Exception as e:
    logger.warning(f"⚠️ Filtrage non supporté par cette version de FAISS: {e}")

print("\n" + "="*60)
print("✅ Tests terminés!")
print("="*60)
