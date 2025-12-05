from src.rag.embedder import Embedder
from config.config import PROCESSED_DATA
import pandas as pd
from config.config import MISTRAL_API_KEY,EMBEDDING_MODEL,EMBEDDINGS_DATA
import time
import json
import logging

df = pd.read_csv(PROCESSED_DATA)

embedder = Embedder(api_key=MISTRAL_API_KEY,model=EMBEDDING_MODEL)
embeddings:list =[]
for idx, row in df.iterrows():
    text = f"{row['title.fr']}. {row['description.fr']}"
    # print(text)
    chunks, vectors = embedder.chunk_and_embed(text=text)

    for chunk, vector in zip(chunks,vectors):
        embeddings.append({'event_id': int(row['event_id']),
                           'chunk':chunk,
                           'vector':vector.tolist(),
                           'title':row['title.fr'],
                           'keywords':row['keywords.fr'],
                           'start_date':row['firstTiming.begin'],
                           'end_date':row['lastTiming.end'],
                           'location_adress':row['location.address'],
                           'location_name':row['location.name']})
    
    print(f"Evenement n°{idx} a été traité")

with open(EMBEDDINGS_DATA,'w',encoding='utf-8') as f :
    json.dump(embeddings,f,ensure_ascii=False,indent=2)