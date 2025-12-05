from src.rag.embedder import Embedder
from config.config import PROCESSED_DATA
import pandas as pd
from config.config import MISTRAL_API_KEY,EMBEDDING_MODEL,EMBEDDINGS_DATA
import time
import json
from tqdm import tqdm

df = pd.read_csv(PROCESSED_DATA)

embedder = Embedder(api_key=MISTRAL_API_KEY,model=EMBEDDING_MODEL)
embeddings:list =[]

for idx, row in tqdm(enumerate(df.iterrows()), total=len(df), desc="Traitement des événements"):
    text = f"{row[1]['title.fr']}. {row[1]['description.fr']}"
    chunks, vectors = embedder.chunk_and_embed(text=text)

    for chunk, vector in zip(chunks, vectors):
        embeddings.append({
            'event_id': int(row[1]['event_id']),
            'chunk': chunk,
            'vector': vector.tolist(),
            'title': row[1]['title.fr'],
            'keywords': row[1]['keywords.fr'],
            'start_date': row[1]['firstTiming.begin'],
            'end_date': row[1]['lastTiming.end'],
            'location_adress': row[1]['location.address'],
            'location_name': row[1]['location.name']
        })

    
    # print(f"Evenement n°{idx} a été traité")

with open(EMBEDDINGS_DATA,'w',encoding='utf-8') as f :
    json.dump(embeddings,f,ensure_ascii=False,indent=2)