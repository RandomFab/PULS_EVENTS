"""
Additional tests for EmbeddingService.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
import tempfile
from pathlib import Path
from src.domain.services.embedding_service import EmbeddingService


class TestEmbeddingServiceBatchProcessing:
    """Tests du traitement par batch"""
    
    def test_batch_size_attribute(self):
        """Vérifie l'attribut batch_size"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_embedder = Mock()
            service = EmbeddingService(
                embedder=mock_embedder,
                embeddings_path=Path(tmpdir) / 'emb.json',
                processed_data_path=Path(tmpdir) / 'proc.csv'
            )
            
            assert hasattr(service, 'batch_size')
            assert service.batch_size > 0


class TestEmbeddingServiceEdgeCases:
    """Tests des cas limites"""
    
    def test_empty_dataframe(self):
        """Vérifie avec un DataFrame vide"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_embedder = Mock()
            service = EmbeddingService(
                embedder=mock_embedder,
                embeddings_path=Path(tmpdir) / 'emb.json',
                processed_data_path=Path(tmpdir) / 'proc.csv'
            )
            
            empty_df = pd.DataFrame()
            chunks = service._create_chunks(empty_df)
            
            assert isinstance(chunks, list)
    
    def test_safe_value_with_string(self):
        """Vérifie safe_value avec une chaîne"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_embedder = Mock()
            service = EmbeddingService(
                embedder=mock_embedder,
                embeddings_path=Path(tmpdir) / 'emb.json',
                processed_data_path=Path(tmpdir) / 'proc.csv'
            )
            
            result = service._safe_value('texte normal')
            assert result == 'texte normal'


class TestEmbeddingServicePaths:
    """Tests des chemins"""
    
    def test_embeddings_path_exists(self):
        """Vérifie l'attribut embeddings_path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_embedder = Mock()
            emb_path = Path(tmpdir) / 'embeddings.json'
            
            service = EmbeddingService(
                embedder=mock_embedder,
                embeddings_path=emb_path,
                processed_data_path=Path(tmpdir) / 'proc.csv'
            )
            
            assert service.embeddings_path == emb_path
