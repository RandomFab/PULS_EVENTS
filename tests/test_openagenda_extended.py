"""
Additional tests for OpenAgendaClient.
"""
import pytest
from unittest.mock import Mock, patch
import requests


class TestOpenAgendaClientErrorHandling:
    """Tests de gestion des erreurs"""
    
    @patch('src.infrastructure.openagenda_client.requests.get')
    def test_get_events_http_error(self, mock_get):
        """Vérifie la gestion des erreurs HTTP"""
        from src.infrastructure.openagenda_client import OpenAgendaClient
        
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.HTTPError("Bad request")
        mock_get.return_value = mock_response
        
        client = OpenAgendaClient(api_key="test-key")
        
        with pytest.raises(requests.HTTPError):
            client.get_events(
                agendaUID='20500020',
                start_date='01/01/25',
                end_date='31/01/25'
            )


class TestOpenAgendaClientParameters:
    """Tests des paramètres"""
    
    @patch('src.infrastructure.openagenda_client.requests.get')
    def test_with_additional_params(self, mock_get):
        """Vérifie avec des paramètres supplémentaires"""
        from src.infrastructure.openagenda_client import OpenAgendaClient
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response
        
        client = OpenAgendaClient(api_key="test-key")
        client.get_events(
            agendaUID='20500020',
            start_date='01/01/25',
            end_date='31/01/25',
            params={'q': 'concert'}
        )
        
        assert mock_get.called
