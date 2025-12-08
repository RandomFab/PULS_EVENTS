import json
import numpy as np
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from config.logger import logger


class LangChainFaissIndexer:
    """
    Indexer FAISS utilisant LangChain pour une meilleure intégration.
    Gère automatiquement les embeddings, métadonnées et recherche sémantique.
    """
    
    def __init__(self, index_dir: str, embedding_client: MistralAIEmbeddings):
        """
        Args:
            index_dir: Répertoire pour sauvegarder/charger l'index
            embedding_client: Instance de MistralAIEmbeddings (déjà initialisée)
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_client = embedding_client
        self.vectorstore = None
    
    def build_from_embeddings_file(self, embeddings_path: str):
        """
        Construit le vectorstore à partir d'un fichier JSON d'embeddings.
        
        Args:
            embeddings_path: Chemin vers le fichier embeddings.json
        """
        try:
            logger.info(f"🔄 Chargement des embeddings depuis {embeddings_path}")
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            logger.info(f"✅ {len(embeddings_data)} embeddings chargés")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des embeddings: {e}")
            raise
        
        # Conversion en Documents LangChain
        documents = []
        vectors = []
        
        logger.info("📦 Conversion en format LangChain...")
        for item in embeddings_data:
            # Création du document avec contenu et métadonnées
            doc = Document(
                page_content=item['chunk'],
                metadata={
                    'event_id': item['event_id'],
                    'title': item['title'],
                    'keywords': item['keywords'],
                    'start_date': item['start_date'],
                    'end_date': item['end_date'],
                    'location_address': item['location_address'],
                    'location_name': item['location_name']
                }
            )
            documents.append(doc)
            vectors.append(item['vector'])
        
        vectors = np.array(vectors, dtype='float32')
        
        logger.info(f"🏗️ Construction du FAISS VectorStore (dimension={vectors.shape[1]})...")
        
        # Création du vectorstore avec les vecteurs pré-calculés
        text_embedding_pairs = list(zip([doc.page_content for doc in documents], vectors.tolist()))
        
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embedding_client,
            metadatas=[doc.metadata for doc in documents]
        )
        
        logger.info(f"✅ VectorStore créé avec {len(documents)} documents")
    
    def build_from_documents(self, documents: list[Document]):
        """
        Construit le vectorstore directement depuis des Documents LangChain.
        Les embeddings seront générés automatiquement.
        
        Args:
            documents: Liste de Documents LangChain
        """
        logger.info(f"🧠 Génération des embeddings pour {len(documents)} documents...")
        
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_client
        )
        
        logger.info(f"✅ VectorStore créé avec {len(documents)} documents")
    
    def save(self):
        """Sauvegarde le vectorstore sur le disque."""
        if self.vectorstore is None:
            logger.error("❌ Aucun vectorstore à sauvegarder")
            return
        
        try:
            logger.info(f"💾 Sauvegarde du vectorstore dans {self.index_dir}...")
            self.vectorstore.save_local(str(self.index_dir))
            logger.info("✅ VectorStore sauvegardé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde: {e}")
            raise
    
    def load(self):
        """Charge le vectorstore depuis le disque."""
        try:
            logger.info(f"🔄 Chargement du vectorstore depuis {self.index_dir}...")
            self.vectorstore = FAISS.load_local(
                str(self.index_dir),
                self.embedding_client,
                allow_dangerous_deserialization=True  # Nécessaire pour charger le pickle
            )
            logger.info("✅ VectorStore chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            raise
    
    def search(self, query: str, k: int = 5, score_threshold: float = None):
        """
        Recherche sémantique dans le vectorstore.
        
        Args:
            query: Question ou texte de recherche
            k: Nombre de résultats à retourner
            score_threshold: Seuil de similarité minimum (optionnel)
            
        Returns:
            Liste de tuples (Document, score)
        """
        if self.vectorstore is None:
            logger.error("❌ VectorStore non initialisé. Utilisez load() ou build_*()")
            raise ValueError("VectorStore non initialisé")
        
        logger.info(f"🔍 Recherche: '{query[:50]}...' (top {k})")
        
        if score_threshold:
            # Recherche avec seuil de score
            results = self.vectorstore.similarity_search_with_score(
                query, k=k
            )
            results = [(doc, score) for doc, score in results if score <= score_threshold]
        else:
            # Recherche standard
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        logger.info(f"✅ {len(results)} résultats trouvés")
        return results
    
    def search_by_vector(self, query_vector: np.ndarray, k: int = 5):
        """
        Recherche par vecteur d'embedding directement.
        
        Args:
            query_vector: Vecteur d'embedding (1D ou 2D array)
            k: Nombre de résultats
            
        Returns:
            Liste de tuples (Document, score)
        """
        if self.vectorstore is None:
            logger.error("❌ VectorStore non initialisé")
            raise ValueError("VectorStore non initialisé")
        
        # Assurer que le vecteur est 1D
        if query_vector.ndim == 2:
            query_vector = query_vector[0]
        
        query_vector = query_vector.astype('float32')
        
        logger.info(f"🔍 Recherche par vecteur (dimension={len(query_vector)})")
        
        results = self.vectorstore.similarity_search_by_vector(
            embedding=query_vector.tolist(),
            k=k
        )
        
        # Pour obtenir les scores, on peut utiliser une recherche avec le vecteur
        results_with_scores = []
        for doc in results:
            # Calcul manuel du score si nécessaire
            results_with_scores.append((doc, None))
        
        logger.info(f"✅ {len(results)} résultats trouvés")
        return results_with_scores
    
    def filter_by_metadata(self, query: str, k: int = 5, filter_dict: dict = None):
        """
        Recherche avec filtrage sur les métadonnées.
        
        Args:
            query: Texte de recherche
            k: Nombre de résultats
            filter_dict: Dictionnaire de filtres (ex: {'event_id': 123})
            
        Returns:
            Liste de Documents
        """
        if self.vectorstore is None:
            raise ValueError("VectorStore non initialisé")
        
        logger.info(f"🔍 Recherche avec filtres: {filter_dict}")
        
        # Note: FAISS de LangChain supporte les filtres basiques
        results = self.vectorstore.similarity_search(
            query, 
            k=k,
            filter=filter_dict
        )
        
        logger.info(f"✅ {len(results)} résultats filtrés trouvés")
        return results
    
    def add_documents(self, documents: list[Document]):
        """
        Ajoute de nouveaux documents au vectorstore existant.
        
        Args:
            documents: Liste de Documents à ajouter
        """
        if self.vectorstore is None:
            logger.error("❌ VectorStore non initialisé")
            raise ValueError("VectorStore non initialisé")
        
        logger.info(f"➕ Ajout de {len(documents)} nouveaux documents...")
        self.vectorstore.add_documents(documents)
        logger.info("✅ Documents ajoutés avec succès")
