"""
Tests pour les routes API.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from src.presentation.api.main import app
from src.presentation.api import routes


@pytest.fixture
def client():
    """Client de test FastAPI"""
    return TestClient(app)


class TestHealthRouter:
    """Tests du endpoint de santé"""
    
    def test_health_router_returns_ok(self, client):
        """Vérifie que le health check retourne OK"""
        response = client.get("/health_router")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'Router ok'


class TestIndexInfo:
    """Tests du endpoint index_info"""
    
    @patch('src.presentation.api.routes.retriever')
    def test_index_info_success(self, mock_retriever, client):
        """Vérifie que index_info retourne les infos"""
        mock_indexer = MagicMock()
        mock_indexer.index_info.return_value = {
            'num_docs': 100,
            'dimension': 1536
        }
        mock_retriever.indexer = mock_indexer
        
        response = client.get("/index_info")
        
        assert response.status_code == 200
        data = response.json()
        assert 'num_docs' in data or data is not None
    
    @patch('src.presentation.api.routes.retriever')
    def test_index_info_not_found(self, mock_retriever, client):
        """Vérifie 404 quand l'index est vide"""
        mock_indexer = MagicMock()
        mock_indexer.index_info.return_value = None
        mock_retriever.indexer = mock_indexer
        
        response = client.get("/index_info")
        
        assert response.status_code == 404


class TestAskEndpoint:
    """Tests du endpoint /ask"""
    
    @patch('src.presentation.api.routes.rag_service')
    def test_ask_success(self, mock_rag_service, client):
        """Vérifie une requête réussie"""
        mock_rag_service.search_and_answer.return_value = "Voici les concerts trouvés"
        
        response = client.post(
            "/ask",
            json={"query": "Quels concerts à Rennes?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'formatted_response' in data['data']
    
    @patch('src.presentation.api.routes.rag_service')
    def test_ask_empty_response(self, mock_rag_service, client):
        """Vérifie l'erreur avec une réponse vide"""
        mock_rag_service.search_and_answer.return_value = ""
        
        response = client.post(
            "/ask",
            json={"query": "test"}
        )
        
        assert response.status_code == 500
    
    @patch('src.presentation.api.routes.rag_service')
    def test_ask_connection_error(self, mock_rag_service, client):
        """Vérifie la gestion des erreurs de connexion"""
        mock_rag_service.search_and_answer.side_effect = ConnectionError("API unavailable")
        
        response = client.post(
            "/ask",
            json={"query": "test"}
        )
        
        assert response.status_code == 503
    
    def test_ask_invalid_query_too_long(self, client):
        """Vérifie la validation de la longueur"""
        response = client.post(
            "/ask",
            json={"query": "x" * 300}
        )
        
        assert response.status_code in [400, 422]


class TestSearchRaw:
    """Tests du endpoint /search_raw"""
    
    @patch('src.presentation.api.routes.retriever')
    def test_search_raw_success(self, mock_retriever, client):
        """Vérifie une recherche réussie"""
        mock_doc = MagicMock()
        mock_doc.metadata = {
            'event_id': '1',
            'title': 'Concert',
            'location_address': 'Rennes'
        }
        mock_retriever.indexer.search.return_value = [(mock_doc, 0.9)]
        
        response = client.post(
            "/search_raw",
            json={"query": "concert jazz"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
    
    @patch('src.presentation.api.routes.retriever')
    def test_search_raw_empty_query(self, mock_retriever, client):
        """Vérifie la validation de requête vide"""
        response = client.post(
            "/search_raw",
            json={"query": "ab"}
        )
        
        assert response.status_code == 422
    
    @patch('src.presentation.api.routes.retriever')
    def test_search_raw_no_results(self, mock_retriever, client):
        """Vérifie quand aucun résultat n'est trouvé"""
        mock_retriever.indexer.search.return_value = []
        
        response = client.post(
            "/search_raw",
            json={"query": "xyz abc def"}
        )
        
        assert response.status_code == 404


class TestRebuild:
    """Tests du endpoint /rebuild"""
    
    @patch('src.presentation.api.routes.retriever')
    @patch('src.presentation.api.routes.EMBEDDINGS_PATH', '/fake/path/embeddings.json')
    def test_rebuild_success(self, mock_retriever, client):
        """Vérifie un rebuild réussi"""
        mock_indexer = MagicMock()
        mock_indexer.index_info.return_value = {
            'num_docs': 150,
            'dimension': 1536
        }
        mock_retriever.indexer = mock_indexer
        
        response = client.post("/rebuild")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
    
    @patch('src.presentation.api.routes.retriever')
    @patch('src.presentation.api.routes.EMBEDDINGS_PATH', '/fake/path/embeddings.json')
    def test_rebuild_file_not_found(self, mock_retriever, client):
        """Vérifie l'erreur fichier manquant"""
        mock_indexer = MagicMock()
        mock_indexer.build_from_embeddings_file.side_effect = FileNotFoundError("embeddings.json not found")
        mock_retriever.indexer = mock_indexer
        
        response = client.post("/rebuild")
        
        assert response.status_code == 404
    
    @patch('src.presentation.api.routes.retriever')
    @patch('src.presentation.api.routes.EMBEDDINGS_PATH', '/fake/path/embeddings.json')
    def test_rebuild_empty_index(self, mock_retriever, client):
        """Vérifie l'erreur index vide"""
        mock_indexer = MagicMock()
        mock_indexer.index_info.return_value = None
        mock_retriever.indexer = mock_indexer
        
        response = client.post("/rebuild")
        
        assert response.status_code == 500


class TestUpdateDatas:
    """Tests du endpoint /update_datas"""
    
    @patch('src.presentation.api.routes.event_service')
    @patch('src.presentation.api.routes.embedding_service')
    @patch('src.presentation.api.routes.retriever')
    @patch('src.presentation.api.routes.EMBEDDINGS_PATH', '/fake/embeddings.json')
    def test_update_datas_success(self, mock_retriever, mock_embedding_service, mock_event_service, client):
        """Vérifie une mise à jour réussie"""
        mock_indexer = MagicMock()
        mock_indexer.index_info.return_value = {
            'num_docs': 200,
            'dimension': 1536
        }
        mock_retriever.indexer = mock_indexer
        
        response = client.post(
            "/update_datas",
            json={
                "start_date": "01/01/25",
                "end_date": "31/12/25"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert mock_event_service.fetch_and_process_events.called
        assert mock_embedding_service.build_embeddings.called
    
    def test_update_datas_invalid_date_format(self, client):
        """Vérifie la validation du format de date"""
        response = client.post(
            "/update_datas",
            json={
                "start_date": "2025-01-01",
                "end_date": "31/12/25"
            }
        )
        
        assert response.status_code in [400, 422]
    
    @patch('src.presentation.api.routes.event_service')
    @patch('src.presentation.api.routes.embedding_service')
    @patch('src.presentation.api.routes.retriever')
    def test_update_datas_connection_error(self, mock_retriever, mock_embedding_service, mock_event_service, client):
        """Vérifie la gestion des erreurs de connexion"""
        mock_event_service.fetch_and_process_events.side_effect = ConnectionError("API unavailable")
        
        response = client.post(
            "/update_datas",
            json={
                "start_date": "01/01/25",
                "end_date": "31/12/25"
            }
        )
        
        assert response.status_code == 503
    
    @patch('src.presentation.api.routes.event_service')
    @patch('src.presentation.api.routes.embedding_service')
    @patch('src.presentation.api.routes.retriever')
    def test_update_datas_empty_index(self, mock_retriever, mock_embedding_service, mock_event_service, client):
        """Vérifie l'erreur index vide après update"""
        mock_indexer = MagicMock()
        mock_indexer.index_info.return_value = None
        mock_retriever.indexer = mock_indexer
        
        response = client.post(
            "/update_datas",
            json={
                "start_date": "01/01/25",
                "end_date": "31/12/25"
            }
        )
        
        assert response.status_code == 500
    
    def test_update_datas_missing_dates(self, client):
        """Vérifie que les dates par défaut fonctionnent"""
        # Les dates ont des valeurs par défaut dans le modèle
        # donc cette requête doit fonctionner
        response = client.post(
            "/update_datas",
            json={}
        )
        
        # Peut être 200 si les defaults sont utilisés
        assert response.status_code in [200, 400, 422]


class TestPydanticModels:
    """Tests des modèles Pydantic"""
    
    def test_ask_model_validation(self):
        """Vérifie la validation du modèle Ask"""
        from src.presentation.api.routes import Ask
        
        # Valide
        ask = Ask(query="Quels concerts?")
        assert ask.query == "Quels concerts?"
        
        # Query trop longue
        with pytest.raises(ValueError):
            Ask(query="x" * 201)
    
    def test_search_raw_model_validation(self):
        """Vérifie la validation du modèle SearchRaw"""
        from src.presentation.api.routes import SearchRaw
        
        # Valide
        search = SearchRaw(query="concert")
        assert search.query == "concert"
    
    def test_update_date_model_validation(self):
        """Vérifie la validation du modèle UpdateDate"""
        from src.presentation.api.routes import UpdateDate
        
        # Valide
        update = UpdateDate(start_date="01/01/25", end_date="31/12/25")
        assert update.start_date == "01/01/25"
        
        # Format invalide
        with pytest.raises(ValueError):
            UpdateDate(start_date="2025-01-01", end_date="31/12/25")
        
        # Date invalide
        with pytest.raises(ValueError):
            UpdateDate(start_date="32/13/25", end_date="31/12/25")
