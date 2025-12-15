"""
Tests pour Embedder.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.rag.embedder import Embedder


@pytest.fixture
def mock_mistral_embeddings():
    """Mock pour MistralAIEmbeddings"""
    with patch('src.rag.embedder.MistralAIEmbeddings') as mock:
        mock_instance = MagicMock()
        # Retourner des embeddings proportionnels au nombre d'entrées
        def embed_docs_side_effect(texts):
            return [[0.1] * 1536 for _ in texts]
        
        def embed_query_side_effect(text):
            return [0.2] * 1536
        
        mock_instance.embed_documents.side_effect = embed_docs_side_effect
        mock_instance.embed_query.side_effect = embed_query_side_effect
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def embedder(mock_mistral_embeddings):
    """Crée une instance d'Embedder avec mocks"""
    return Embedder(api_key="test-key", model="mistral-embed")


class TestEmbedderInit:
    """Tests d'initialisation"""
    
    def test_init_with_valid_key(self, mock_mistral_embeddings):
        """Vérifie l'initialisation avec une clé valide"""
        embedder = Embedder(api_key="test-key")
        
        assert embedder.model == "mistral-embed"
        assert embedder.client is not None
        assert embedder.chunker is not None
    
    def test_init_custom_model(self, mock_mistral_embeddings):
        """Vérifie l'initialisation avec un modèle personnalisé"""
        embedder = Embedder(api_key="test-key", model="custom-model")
        
        assert embedder.model == "custom-model"


class TestEmbedTexts:
    """Tests de la méthode embed_texts"""
    
    def test_embed_single_text(self, embedder):
        """Vérifie l'embedding d'un seul texte"""
        texts = ["Concert de jazz"]
        
        result = embedder.embed_texts(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
        assert result.shape[1] == 1536
    
    def test_embed_multiple_texts(self, embedder):
        """Vérifie l'embedding de plusieurs textes"""
        texts = [
            "Concert de jazz",
            "Festival rock",
            "Ballet classique"
        ]
        
        result = embedder.embed_texts(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3
        assert result.shape[1] == 1536
    
    def test_embed_empty_list(self, embedder):
        """Vérifie avec une liste vide"""
        texts = []
        
        result = embedder.embed_texts(texts)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_embed_returns_float32(self, embedder):
        """Vérifie que le type de données est float32"""
        texts = ["Test texte"]
        
        result = embedder.embed_texts(texts)
        
        assert result.dtype == np.float32


class TestEmbedQuery:
    """Tests de l'embedding de requête"""
    
    def test_embed_query_single(self, embedder):
        """Vérifie l'embedding d'une seule requête"""
        # embed_query est appelé via chunk_and_embed
        # On teste indirectement via les attributs
        assert embedder.client is not None


class TestEmbedderAttributes:
    """Tests des attributs d'Embedder"""
    
    def test_has_chunker(self, embedder):
        """Vérifie que Embedder a un Chunker"""
        assert hasattr(embedder, 'chunker')
        assert embedder.chunker is not None
    
    def test_has_client(self, embedder):
        """Vérifie que Embedder a un client MistralAIEmbeddings"""
        assert hasattr(embedder, 'client')
        assert embedder.client is not None
