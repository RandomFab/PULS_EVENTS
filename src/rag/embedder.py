from src.rag.chunker import Chunker
from mistralai.client import MistralClient
import numpy as np

class Embedder:
    def __init__(self,api_key:str,model:str='mistral-embed'):
        self.client: MistralClient = MistralClient(api_key=api_key)
        self.model:str = model
        self.chunker:Chunker = Chunker()

    def chunk_and_embed(self,text):

        chunks:list[str]=[]

        chunks = self.chunker.split_text(text)

        vectors = self.client.embeddings(model=self.model, input_texts=chunks).vectors

        vectors = np.array(vectors).astype("float32")

        return chunks,vectors
