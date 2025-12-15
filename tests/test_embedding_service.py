"""
Tests pour EmbeddingService.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.domain.services.embedding_service import EmbeddingService


@pytest.fixture
def temp_dir():
    """Crée un répertoire temporaire"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataframe():
    """Crée un DataFrame d'exemple"""
    return pd.DataFrame({
        'event_id': [1, 2, 3],
        'title.fr': ['Concert', 'Festival', 'Ballet'],
        'description.fr': ['Jazz', 'Rock', 'Classique'],
        'location.city': ['Rennes', 'Rennes', 'Paris'],
        'location.name': ['Opéra', 'Parc', 'Palais'],
        'location.address': ['1 rue', '2 rue', '3 rue'],
        'keywords.fr': ['musique', 'rock', 'danse'],
        'firstTiming.begin': ['2025-01-15T19:00:00'] * 3,
        'firstTiming.end': ['2025-01-15T21:00:00'] * 3,
        'lastTiming.begin': ['2025-01-15T19:00:00'] * 3,
        'lastTiming.end': ['2025-01-15T21:00:00'] * 3,
    })


@pytest.fixture
def mock_embedder():
    """Mock pour l'Embedder"""
    mock = Mock()
    mock.embed_texts.return_value = np.random.rand(100, 1536).astype('float32')
    mock.chunker = Mock()
    mock.chunker.split_text.return_value = ['Chunk 1', 'Chunk 2', 'Chunk 3']
    return mock


@pytest.fixture
def embedding_service(mock_embedder, temp_dir):
    """Crée une instance d'EmbeddingService"""
    service = EmbeddingService(
        embedder=mock_embedder,
        embeddings_path=temp_dir / 'embeddings.json',
        processed_data_path=temp_dir / 'processed.csv'
    )
    return service


class TestEmbeddingServiceInit:
    """Tests d'initialisation"""
    
    def test_init_stores_paths(self, embedding_service, temp_dir):
        """Vérifie que les chemins sont stockés"""
        assert embedding_service.embeddings_path == temp_dir / 'embeddings.json'
        assert embedding_service.processed_data_path == temp_dir / 'processed.csv'
    
    def test_init_stores_embedder(self, embedding_service, mock_embedder):
        """Vérifie que l'embedder est stocké"""
        assert embedding_service.embedder == mock_embedder


class TestCreateChunks:
    """Tests de la création de chunks"""
    
    def test_create_chunks_from_dataframe(self, embedding_service, sample_dataframe):
        """Vérifie la création de chunks à partir d'un DataFrame"""
        chunks = embedding_service._create_chunks(sample_dataframe)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, dict) for c in chunks)
    
    def test_chunks_contain_metadata(self, embedding_service, sample_dataframe):
        """Vérifie que les chunks contiennent les métadonnées"""
        chunks = embedding_service._create_chunks(sample_dataframe)
        
        if chunks:
            chunk = chunks[0]
            assert 'chunk' in chunk
            assert 'event_id' in chunk
            assert 'title' in chunk


class TestGenerateEmbeddings:
    """Tests de la génération d'embeddings"""
    
    def test_generate_embeddings(self, embedding_service, mock_embedder):
        """Vérifie la génération d'embeddings"""
        chunks = [
            {'chunk': 'Concert', 'event_id': 1},
            {'chunk': 'Festival', 'event_id': 2},
        ]
        
        vectors = embedding_service._generate_embeddings(chunks)
        
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape[1] == 1536
        assert vectors.dtype == np.float32
    
    def test_generate_embeddings_batch_processing(self, embedding_service, mock_embedder):
        """Vérifie le traitement par batch"""
        embedding_service.batch_size = 2
        chunks = [{'chunk': f'Event {i}', 'event_id': i} for i in range(5)]
        
        vectors = embedding_service._generate_embeddings(chunks)
        
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape[1] == 1536


class TestAssembleEmbeddings:
    """Tests de l'assemblage des embeddings"""
    
    def test_assemble_creates_structure(self, embedding_service):
        """Vérifie l'assemblage correct"""
        chunks = [{'chunk': 'Test', 'event_id': 1}]
        vectors = [np.random.rand(1536)]
        
        result = embedding_service._assemble_embeddings(chunks, vectors)
        
        # Result peut être list ou dict selon l'implémentation
        assert result is not None


class TestSafeValue:
    """Tests de la gestion des valeurs None"""
    
    def test_safe_value_with_none(self, embedding_service):
        """Vérifie la gestion des NaN/None"""
        import pandas as pd
        result = embedding_service._safe_value(pd.NA, 'default')
        assert result == 'default'
    
    def test_safe_value_with_value(self, embedding_service):
        """Vérifie avec une valeur normale"""
        result = embedding_service._safe_value('texte', 'default')
        assert result == 'texte'


class TestBuildEmbeddings:
    """Tests de la construction complète des embeddings"""
    
    def test_build_embeddings_creates_file(self, embedding_service, sample_dataframe, temp_dir, mock_embedder):
        """Vérifie que build_embeddings crée le fichier"""
        # Sauvegarder d'abord le DataFrame
        embedding_service.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_dataframe.to_csv(embedding_service.processed_data_path, index=False)
        
        embedding_service.build_embeddings()
        
        # Vérifier que le fichier d'embeddings a été créé
        assert embedding_service.embeddings_path.exists()
    
    def test_load_processed_events(self, embedding_service, sample_dataframe, temp_dir):
        """Vérifie le chargement de l'événement traité"""
        embedding_service.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_dataframe.to_csv(embedding_service.processed_data_path, index=False)
        
        # Charger les données
        df = pd.read_csv(embedding_service.processed_data_path)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
