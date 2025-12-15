"""
Service de génération d'embeddings.
Logique métier pour créer et gérer les embeddings des événements.
"""
from pathlib import Path
import pandas as pd
import json
import numpy as np
import time
from tqdm import tqdm
from config.logger import logger


class EmbeddingService:
    """Service métier pour la génération d'embeddings"""
    
    def __init__(
        self, 
        embedder,
        processed_data_path: Path,
        embeddings_path: Path,
        batch_size: int = 10
    ):
        """
        Initialise le service d'embeddings.
        
        Args:
            embedder: Instance de la classe Embedder (RAG layer)
            processed_data_path: Chemin vers les données d'événements traitées
            embeddings_path: Chemin de sauvegarde des embeddings
            batch_size: Taille des batches pour l'API (défaut: 10)
        """
        self.embedder = embedder
        self.processed_data_path = processed_data_path
        self.embeddings_path = embeddings_path
        self.batch_size = batch_size
    
    def build_embeddings(self) -> None:
        """
        Génère les embeddings des événements et les sauvegarde dans un fichier JSON.
        
        Cette fonction :
        1. Charge les événements depuis le CSV traité
        2. Découpe chaque événement en chunks avec métadonnées
        3. Génère les embeddings en batch via l'API Mistral
        4. Sauvegarde le résultat dans embeddings.json
        
        Raises:
            FileNotFoundError: Si le fichier processed_data_path n'existe pas
            Exception: Pour les erreurs d'API ou d'écriture fichier
        """
        # Chargement des données
        df = pd.read_csv(self.processed_data_path)
        logger.info(f"📂 {len(df)} événements chargés depuis {self.processed_data_path}")

        # 📊 Phase 1 : Préparation des données (chunking)
        logger.info("🔪 Phase 1 : Découpage en chunks...")
        all_chunks_data = self._create_chunks(df)
        logger.info(f"✅ {len(all_chunks_data)} chunks générés pour {len(df)} événements")

        # 🚀 Phase 2 : Embedding en batch avec rate limiting
        logger.info("🧠 Phase 2 : Génération des embeddings en batch...")
        all_vectors = self._generate_embeddings(all_chunks_data)
        logger.info(f"✅ {len(all_vectors)} vecteurs générés")
        
        # 📦 Phase 3 : Assemblage final
        logger.info("📦 Phase 3 : Assemblage des résultats...")
        embeddings = self._assemble_embeddings(all_chunks_data, all_vectors)

        # 💾 Sauvegarde
        logger.info(f"💾 Sauvegarde de {len(embeddings)} embeddings dans {self.embeddings_path}")
        self._save_embeddings(embeddings)

        logger.info("✅ Traitement terminé avec succès!")
    
    def _create_chunks(self, df: pd.DataFrame) -> list[dict]:
        """
        Découpe les événements en chunks avec métadonnées.
        
        Args:
            df: DataFrame des événements
            
        Returns:
            Liste de dictionnaires {chunk, event_id, title, keywords, ...}
        """
        all_chunks_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="📝 Chunking"):
            # Construction du texte structuré pour l'événement avec gestion des NaN
            text = (
                f"Titre : {self._safe_value(row['title.fr'])}.\n"
                f"Description : {self._safe_value(row['description.fr'], 'Non disponible')}.\n"
                f"Ville : {self._safe_value(row['location.city'], 'Rennes')}.\n"
                f"Adresse : {self._safe_value(row['location.address'], 'Adresse non précisée')}.\n"
                f"Debut : {self._safe_value(row['firstTiming.begin'], 'Date non précisée')}.\n"
                f"Fin : {self._safe_value(row['lastTiming.end'], 'Date non précisée')}."
            )
            chunks = self.embedder.chunker.split_text(text)
        
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
                    'location_name': row['location.name'],
                    'location_city': row['location.city']
                })

        return all_chunks_data
    
    def _generate_embeddings(self, all_chunks_data: list[dict]) -> np.ndarray:
        """
        Génère les embeddings en batch avec gestion des erreurs et rate limiting.
        
        Args:
            all_chunks_data: Liste des chunks avec métadonnées
            
        Returns:
            Array numpy des vecteurs d'embeddings
        """
        all_chunk_texts = [item['chunk'] for item in all_chunks_data]
        all_vectors = []
        num_batches = (len(all_chunk_texts) + self.batch_size - 1) // self.batch_size

        for i in tqdm(
            range(0, len(all_chunk_texts), self.batch_size),
            total=num_batches,
            desc="🔢 Embedding"
        ):
            batch = all_chunk_texts[i:i + self.batch_size]
            
            try:
                batch_vectors = self.embedder.embed_texts(batch)
                all_vectors.extend(batch_vectors)
                
                # Pause entre les batches (sauf pour le dernier)
                if i + self.batch_size < len(all_chunk_texts):
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"❌ Erreur lors du batch {i//self.batch_size + 1}: {e}")
                logger.info("⏸️ Pause de 3 secondes avant de réessayer...")
                time.sleep(3)
                
                # Retry
                try:
                    batch_vectors = self.embedder.embed_texts(batch)
                    all_vectors.extend(batch_vectors)
                except Exception as retry_error:
                    logger.error(f"❌ Échec après retry: {retry_error}")
                    raise

        return np.array(all_vectors)
    
    def _assemble_embeddings(
        self,
        all_chunks_data: list[dict],
        all_vectors: np.ndarray
    ) -> list[dict]:
        """
        Assemble les chunks et leurs vecteurs d'embeddings.
        
        Args:
            all_chunks_data: Liste des chunks avec métadonnées
            all_vectors: Array numpy des vecteurs
            
        Returns:
            Liste de dictionnaires {chunk, vector, event_id, ...}
        """
        embeddings = []
        for chunk_data, vector in zip(all_chunks_data, all_vectors):
            embeddings.append({
                **chunk_data,  # Copie toutes les métadonnées
                'vector': vector.tolist()
            })
        return embeddings
    
    def _save_embeddings(self, embeddings: list[dict]) -> None:
        """Sauvegarde les embeddings dans un fichier JSON"""
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _safe_value(value, default: str = 'Non renseigné') -> str:
        """
        Retourne la valeur ou un texte par défaut si NaN/None.
        
        Args:
            value: Valeur à vérifier
            default: Valeur par défaut si NaN/None
            
        Returns:
            Valeur nettoyée ou défaut
        """
        if pd.isna(value) or value == '' or str(value).strip() == '':
            return default
        return str(value)
