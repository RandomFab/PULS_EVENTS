"""
Tests pour EventService.
"""
import pytest
import pandas as pd
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.domain.services.event_service import EventService


@pytest.fixture
def temp_paths(tmp_path):
    """Crée des chemins temporaires pour les tests"""
    raw_path = tmp_path / "raw_data.json"
    processed_path = tmp_path / "processed_data.csv"
    return raw_path, processed_path


@pytest.fixture
def mock_openagenda_client():
    """Mock pour le client OpenAgenda"""
    mock = Mock()
    mock.get_events_with_keywords.return_value = {
        'event_1': {
            'event_id': 'event_1',
            'title.fr': 'Concert',
            'description.fr': 'Concert de jazz',
            'location.city': 'Rennes',
            'location.name': 'Opéra',
            'location.address': '1 rue de l\'Opéra',
            'firstTiming.begin': '2025-01-15T19:00:00',
            'firstTiming.end': '2025-01-15T21:00:00',
            'lastTiming.begin': '2025-01-15T19:00:00',
            'lastTiming.end': '2025-01-15T21:00:00',
            'keywords.fr': 'musique,concert',
            'dateRange.fr': '15/01/2025',
            'originAgenda.title': 'Rennes'
        }
    }
    return mock


@pytest.fixture
def event_service(mock_openagenda_client, temp_paths):
    """Crée une instance d'EventService"""
    raw_path, processed_path = temp_paths
    return EventService(
        openagenda_client=mock_openagenda_client,
        raw_data_path=raw_path,
        processed_data_path=processed_path
    )


class TestEventServiceInit:
    """Tests d'initialisation"""
    
    def test_init_creates_service(self, event_service, mock_openagenda_client, temp_paths):
        """Vérifie l'initialisation"""
        assert event_service.client == mock_openagenda_client
        assert event_service.raw_data_path == temp_paths[0]
        assert event_service.processed_data_path == temp_paths[1]


class TestFetchAndProcessEvents:
    """Tests de fetch_and_process_events"""
    
    def test_fetch_calls_client(self, event_service, mock_openagenda_client):
        """Vérifie que fetch appelle le client"""
        df = event_service.fetch_and_process_events(
            start_date='01/01/25',
            end_date='31/01/25',
            agenda_uid='20500020'
        )
        
        assert mock_openagenda_client.get_events_with_keywords.called
        assert isinstance(df, pd.DataFrame)
    
    def test_fetch_saves_raw_data(self, event_service, temp_paths):
        """Vérifie que les données brutes sont sauvegardées"""
        raw_path = temp_paths[0]
        
        event_service.fetch_and_process_events()
        
        assert raw_path.exists()
    
    def test_fetch_saves_processed_data(self, event_service, temp_paths):
        """Vérifie que les données traitées sont sauvegardées"""
        processed_path = temp_paths[1]
        
        event_service.fetch_and_process_events()
        
        assert processed_path.exists()
    
    def test_fetch_returns_dataframe(self, event_service):
        """Vérifie le type de retour"""
        df = event_service.fetch_and_process_events()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestLoadProcessedEvents:
    """Tests de load_processed_events"""
    
    def test_load_existing_file(self, event_service):
        """Vérifie le chargement d'un fichier existant"""
        # Créer d'abord les données
        event_service.fetch_and_process_events()
        
        # Charger les données
        df = event_service.load_processed_events()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_load_missing_file_raises_error(self, event_service):
        """Vérifie l'exception si le fichier n'existe pas"""
        with pytest.raises(FileNotFoundError):
            event_service.load_processed_events()


class TestValidateDataframe:
    """Tests de validation"""
    
    def test_validate_with_valid_data(self, event_service):
        """Vérifie la validation avec des données valides"""
        df = event_service.fetch_and_process_events()
        
        # Ne doit pas lever d'exception
        event_service._validate_dataframe(df)
    
    def test_validate_required_columns(self, event_service):
        """Vérifie la présence des colonnes requises"""
        df = event_service.fetch_and_process_events()
        
        # Vérifier que toutes les colonnes requises sont présentes
        for col in ['event_id', 'title.fr', 'location.city']:
            assert col in df.columns


class TestTransformToDataframe:
    """Tests de transformation des données"""
    
    def test_transform_response_to_dataframe(self, event_service):
        """Vérifie la transformation de réponse API en DataFrame"""
        response = {
            'event_1': {
                'event_id': 'event_1',
                'title.fr': 'Concert',
                'description.fr': 'Concert de jazz',
                'location.city': 'Rennes',
                'location.name': 'Opéra',
                'location.address': '1 rue',
                'firstTiming.begin': '2025-01-15T19:00:00',
                'firstTiming.end': '2025-01-15T21:00:00',
                'lastTiming.begin': '2025-01-15T19:00:00',
                'lastTiming.end': '2025-01-15T21:00:00',
                'keywords.fr': 'musique',
                'dateRange.fr': '15/01/2025',
                'originAgenda.title': 'Rennes'
            }
        }
        
        df = event_service._transform_to_dataframe(response)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]['title.fr'] == 'Concert'
