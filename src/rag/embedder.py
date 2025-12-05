from src.rag.chunker import Chunker
from mistralai import Mistral
import numpy as np
import logging
import time

class Embedder:
    def __init__(self,api_key:str,model:str='mistral-embed',batch_size:int=64,max_retries:int=5):
        self.client: Mistral = Mistral(api_key=api_key)
        self.model:str = model
        self.chunker:Chunker = Chunker()

        self.batch_size:int = batch_size
        self.max_retries:int = max_retries

    def _embedded_batch(self,texts):
        last_error = None
        for atempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(model=self.model, inputs=texts)
                return [item.embedding for item in response.data]
            except Exception as e:
                last_error = e
                message = str(e).lower()

                retryable = (
                    "429" in message
                    or "capacity" in message
                    or "rate limit" in message
                    or "server error" in message
                    or "5" == message[:1]   # 5xx
                )

                if not retryable:
                    raise

                logging.warning(f"⏳ API saturée : {last_error}")
                time.sleep(5)

        raise last_error

    def embed_texts(self,texts:list[str]):
        vectors =  []
        for i in range(0,len(texts),self.batch_size):
            batch = texts[i:i + self.batch_size]
            logging.info(f"Envoie d'un batch de {len(batch)} chunks")
            batch_vectors = self._embedded_batch(batch)
            vectors.extend(batch_vectors)
        
        return np.array(vectors, dtype="float32")

    def chunk_and_embed(self,text):

        chunks:list[str]=[]

        chunks = self.chunker.split_text(text)
        vectors = self.embed_texts(chunks)

        return chunks,vectors
