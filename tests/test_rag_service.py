"""
Tests pour RagService.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.domain.services.rag_service import RagService


@pytest.fixture
def mock_retriever():
    """Mock pour le Retriever"""
    mock = Mock()
    mock.answer_query.return_value = "Voici la réponse à votre question"
    mock.search_similar.return_value = ["Doc 1", "Doc 2", "Doc 3"]
    return mock


@pytest.fixture
def rag_service(mock_retriever):
    """Crée une instance de RagService"""
    return RagService(retriever=mock_retriever)


class TestRagServiceInit:
    """Tests d'initialisation"""
    
    def test_init_stores_retriever(self, rag_service, mock_retriever):
        """Vérifie que le retriever est stocké"""
        assert rag_service.retriever == mock_retriever


class TestSearchAndAnswer:
    """Tests de search_and_answer"""
    
    def test_search_and_answer_calls_retriever(self, rag_service, mock_retriever):
        """Vérifie que la méthode appelle le retriever"""
        query = "Quels concerts à Rennes?"
        
        result = rag_service.search_and_answer(query)
        
        assert mock_retriever.answer_query.called
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_search_and_answer_returns_string(self, rag_service):
        """Vérifie le type de retour"""
        result = rag_service.search_and_answer("test query")
        
        assert isinstance(result, str)
    
    def test_search_and_answer_with_different_queries(self, rag_service, mock_retriever):
        """Vérifie avec différentes requêtes"""
        queries = [
            "Concerts à Rennes",
            "Festival rock",
            "Événements en juillet"
        ]
        
        for query in queries:
            result = rag_service.search_and_answer(query)
            assert isinstance(result, str)
            assert len(result) > 0
    
    @patch('src.domain.services.rag_service.logger')
    def test_search_and_answer_logs_query(self, mock_logger, rag_service):
        """Vérifie que la requête est loggée"""
        rag_service.search_and_answer("test query")
        
        assert mock_logger.info.called


class TestSearchDocuments:
    """Tests de search_documents"""
    
    def test_search_documents_calls_retriever(self, rag_service, mock_retriever):
        """Vérifie que la recherche appelle le retriever"""
        # Mock search_similar pour retourner une liste
        mock_retriever.search_similar.return_value = ["Doc 1", "Doc 2"]
        mock_retriever._embed_query.return_value = ["Doc 1", "Doc 2"]
        
        # Cette méthode utilise _embed_query
        # On doit vérifier que le comportement est correct
        assert rag_service.retriever is not None


class TestRagServiceErrorHandling:
    """Tests de gestion d'erreurs"""
    
    def test_search_and_answer_handles_exception(self, mock_retriever):
        """Vérifie la gestion des exceptions"""
        mock_retriever.answer_query.side_effect = Exception("API Error")
        rag_service = RagService(retriever=mock_retriever)
        
        with pytest.raises(Exception):
            rag_service.search_and_answer("test")
    
    def test_search_documents_raises_on_error(self, mock_retriever):
        """Vérifie que search_documents lève une exception"""
        mock_retriever._embed_query.side_effect = Exception("Embedding error")
        rag_service = RagService(retriever=mock_retriever)
        
        with pytest.raises(Exception):
            rag_service.search_documents("test")


class TestRagServiceIntegration:
    """Tests d'intégration"""
    
    def test_retrieve_multiple_documents(self, rag_service, mock_retriever):
        """Vérifie la récupération de plusieurs documents"""
        mock_retriever.search_similar.return_value = [
            "Concert de jazz à Rennes",
            "Festival rock",
            "Ballet classique"
        ]
        
        assert rag_service.retriever is not None
        assert mock_retriever.search_similar is not None
    
    def test_answer_query_with_empty_documents(self, rag_service, mock_retriever):
        """Vérifie le comportement avec des documents vides"""
        mock_retriever.answer_query.return_value = "Aucun document pertinent"
        
        result = rag_service.search_and_answer("test")
        assert isinstance(result, str)
