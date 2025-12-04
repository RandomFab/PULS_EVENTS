from src.rag.embedder import Embedder

import pandas as pd
from config.config import MISTRAL_API_KEY,MODEL_NAME

df = pd.read_csv('.../Data/processed/events_musique_35')

embedder = Embedder(api_key=MISTRAL_API_KEY,model=MODEL_NAME)

for idx, row in df.iterrows():
    text = f'{row['title.fr']}. {row['description.fr']}'

    chunks, vectors = embedder.chunk_and_embed(text=text)
    