"""
Tests pour OpenAgendaClient.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from src.infrastructure.openagenda_client import OpenAgendaClient


@pytest.fixture
def client():
    """Crée une instance d'OpenAgendaClient"""
    return OpenAgendaClient(api_key="test-api-key")


class TestOpenAgendaClientInit:
    """Tests d'initialisation"""
    
    def test_init_stores_api_key(self):
        """Vérifie que la clé API est stockée"""
        client = OpenAgendaClient(api_key="my-api-key")
        assert client.api_key == "my-api-key"
    
    def test_init_has_base_url(self):
        """Vérifie la présence de l'URL de base"""
        client = OpenAgendaClient(api_key="test")
        assert client.BASE_URL == 'https://api.openagenda.com/v2/agendas'


class TestGetEvents:
    """Tests de la méthode get_events"""
    
    @patch('src.infrastructure.openagenda_client.requests.get')
    def test_get_events_success(self, mock_get, client):
        """Vérifie une requête réussie"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'events': [{'id': '1', 'title': 'Concert'}]}
        mock_get.return_value = mock_response
        
        result = client.get_events(
            agendaUID='20500020',
            start_date='01/01/25',
            end_date='31/01/25'
        )
        
        assert result == {'events': [{'id': '1', 'title': 'Concert'}]}
        assert mock_get.called
    
    @patch('src.infrastructure.openagenda_client.requests.get')
    def test_get_events_with_params(self, mock_get, client):
        """Vérifie l'ajout de paramètres supplémentaires"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        client.get_events(
            agendaUID='20500020',
            start_date='01/01/25',
            end_date='31/01/25',
            params={'limit': 100}
        )
        
        # Vérifier que le paramètre limit a été passé
        call_args = mock_get.call_args
        assert call_args[1]['params']['limit'] == 100
    
    @patch('src.infrastructure.openagenda_client.requests.get')
    def test_get_events_date_conversion(self, mock_get, client):
        """Vérifie la conversion des dates en ISO 8601"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        client.get_events(
            agendaUID='20500020',
            start_date='15/01/25',
            end_date='31/01/25'
        )
        
        # Vérifier les paramètres de la requête
        call_args = mock_get.call_args
        params = call_args[1]['params']
        
        assert 'timings[gte]' in params
        assert 'timings[lte]' in params
        assert '2025-01-15' in params['timings[gte]']
        assert '2025-01-31' in params['timings[lte]']
    
    @patch('src.infrastructure.openagenda_client.requests.get')
    def test_get_events_sets_headers(self, mock_get, client):
        """Vérifie la présence du header Authorization"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        client.get_events(
            agendaUID='20500020',
            start_date='01/01/25',
            end_date='31/01/25'
        )
        
        # Vérifier les headers
        call_args = mock_get.call_args
        headers = call_args[1]['headers']
        
        assert 'Authorization' in headers
        assert 'Bearer test-api-key' in headers['Authorization']


class TestGetEventsWithKeywords:
    """Tests de get_events_with_keywords"""
    
    @patch('src.infrastructure.openagenda_client.requests.get')
    def test_get_events_with_keywords(self, mock_get, client):
        """Vérifie la récupération avec keywords"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'event_1': {}}
        mock_get.return_value = mock_response
        
        result = client.get_events_with_keywords(
            agendaUID='20500020',
            start_date='01/01/25',
            end_date='31/01/25',
            keywords=['concert', 'musique']
        )
        
        assert isinstance(result, dict)


class TestBaseUrl:
    """Tests de l'URL de base"""
    
    def test_base_url_correct(self):
        """Vérifie que l'URL de base est correcte"""
        assert OpenAgendaClient.BASE_URL == 'https://api.openagenda.com/v2/agendas'
