from src.rag.embedder import Embedder
from config.config import PROCESSED_DATA, MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDINGS_PATH,EMBEDDING_BATCH_SIZE
from config.logger import logger
import pandas as pd
import json
import numpy as np
import time
from tqdm import tqdm

# Chargement des données
df = pd.read_csv(PROCESSED_DATA)
logger.info(f"📂 {len(df)} événements chargés depuis {PROCESSED_DATA}")

# Initialisation de l'embedder
embedder = Embedder(api_key=MISTRAL_API_KEY, model=EMBEDDING_MODEL)

# 📊 Phase 1 : Préparation des données (chunking)
logger.info("🔪 Phase 1 : Découpage en chunks...")
all_chunks_data = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="📝 Chunking"):
    # Construction du texte structuré pour l'événement
    text = (
        f"Titre : {row['title.fr']}.\n"
        f"Description : {row['description.fr']}.\n"
        f"Ville : {row['location.city']}.\n"
        f"Adresse : {row['location.address']}.\n"
        f"Debut : {row['firstTiming.begin']}.\n"
        f"Fin : {row['lastTiming.end']}."
    )
    
    # Découpage en chunks
    chunks = embedder.chunker.split_text(text)
    
    # Stockage des chunks avec leurs métadonnées
    for chunk in chunks:
        all_chunks_data.append({
            'chunk': chunk,
            'event_id': int(row['event_id']),
            'title': row['title.fr'],
            'keywords': row['keywords.fr'],
            'start_date': row['firstTiming.begin'],
            'end_date': row['lastTiming.end'],
            'location_address': row['location.address'],
            'location_name': row['location.name']
        })

logger.info(f"✅ {len(all_chunks_data)} chunks générés pour {len(df)} événements")

# 🚀 Phase 2 : Embedding en batch avec rate limiting pour API gratuite
logger.info("🧠 Phase 2 : Génération des embeddings en batch...")
all_chunk_texts = [item['chunk'] for item in all_chunks_data]

# Paramètres pour API gratuite Mistral

all_vectors = []
num_batches = (len(all_chunk_texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

for i in tqdm(range(0, len(all_chunk_texts), EMBEDDING_BATCH_SIZE), 
              total=num_batches, 
              desc="🔢 Embedding"):
    batch = all_chunk_texts[i:i + EMBEDDING_BATCH_SIZE]
    
    try:
        batch_vectors = embedder.embed_texts(batch)
        all_vectors.extend(batch_vectors)
        
        # Pause entre les batches (sauf pour le dernier)
        if i + EMBEDDING_BATCH_SIZE < len(all_chunk_texts):
            import time
            time.sleep(1)
            
    except Exception as e:
        logger.error(f"❌ Erreur lors du batch {i//EMBEDDING_BATCH_SIZE + 1}: {e}")
        logger.info("⏸️  Pause de 10 secondes avant de réessayer...")
        import time
        time.sleep(10)
        # Retry
        try:
            batch_vectors = embedder.embed_texts(batch)
            all_vectors.extend(batch_vectors)
        except Exception as retry_error:
            logger.error(f"❌ Échec après retry: {retry_error}")
            raise

all_vectors = np.array(all_vectors)
logger.info(f"✅ {len(all_vectors)} vecteurs générés")

# 📦 Phase 3 : Assemblage final
logger.info("📦 Phase 3 : Assemblage des résultats...")
embeddings = []
for chunk_data, vector in zip(all_chunks_data, all_vectors):
    embeddings.append({
        **chunk_data,  # Copie toutes les métadonnées
        'vector': vector.tolist()
    })

# 💾 Sauvegarde
logger.info(f"💾 Sauvegarde de {len(embeddings)} embeddings dans {EMBEDDINGS_PATH}")
with open(EMBEDDINGS_PATH, 'w', encoding='utf-8') as f:
    json.dump(embeddings, f, ensure_ascii=False, indent=2)

logger.info("✅ Traitement terminé avec succès!")