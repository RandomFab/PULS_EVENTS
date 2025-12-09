from src.rag.chunker import Chunker
from langchain_mistralai import MistralAIEmbeddings
import numpy as np
from config.logger import logger

class Embedder:
    def __init__(self, api_key: str, model: str = 'mistral-embed'):
        """
        Initialise l'embedder avec LangChain.
        LangChain gère automatiquement le batching et les retries.
        
        Args:
            api_key: Clé API Mistral
            model: Nom du modèle d'embedding (défaut: 'mistral-embed')
        """
        self.model: str = model
        self.client: MistralAIEmbeddings = MistralAIEmbeddings(
            api_key=api_key,
            model=model
        )
        self.chunker: Chunker = Chunker()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed une liste de textes.
        LangChain gère automatiquement le batching et les retries.
        
        Args:
            texts: Liste de textes à embedder
            
        Returns:
            Matrice numpy des vecteurs d'embedding
        """
        logger.info(f"📊 Embedding de {len(texts)} chunks...")
        
        try:
            vectors = self.client.embed_documents(texts)
            logger.info(f"✅ {len(vectors)} vecteurs générés")
        except Exception as e:
            logger.error(f"La génération des vecteurs a rencontré un problème : {e}") 
            vectors = []  
        
        return np.array(vectors, dtype="float32")

    def chunk_and_embed(self, text: str):
        """
        Découpe un texte en chunks et génère les embeddings.
        
        Args:
            text: Texte à traiter
            
        Returns:
            Tuple (chunks, vectors)
        """
        chunks = self.chunker.split_text(text)
        
        if not chunks:
            logger.warning("⚠️ Aucun chunk généré")
            return [], np.array([], dtype="float32")
        
        vectors = self.embed_texts(chunks)
        
        return chunks, vectors
