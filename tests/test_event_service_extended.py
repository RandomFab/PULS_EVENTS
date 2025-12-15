"""
Additional tests for EventService.
"""
import pytest
import pandas as pd
import json
from pathlib import Path
from unittest.mock import Mock, patch
from src.domain.services.event_service import EventService


class TestEventServiceDataHandling:
    """Tests de gestion des données"""
    
    def test_required_columns_constant(self):
        """Vérifie la liste des colonnes requises"""
        from src.domain.services.event_service import EventService
        
        assert hasattr(EventService, 'REQUIRED_COLUMNS')
        assert len(EventService.REQUIRED_COLUMNS) > 0


class TestEventServiceIO:
    """Tests d'I/O"""
    
    def test_save_raw_data(self):
        """Vérifie la sauvegarde des données brutes"""
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock
        
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / 'raw.json'
            service = EventService(
                openagenda_client=Mock(),
                raw_data_path=raw_path,
                processed_data_path=Path(tmpdir) / 'processed.csv'
            )
            
            data = {'event_1': {'title': 'Concert'}}
            service._save_raw_data(data)
            
            assert raw_path.exists()


class TestEventServiceDataTransformation:
    """Tests de transformation des données"""
    
    def test_transform_preserves_dataframe_type(self):
        """Vérifie que transform retourne un DataFrame"""
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock
        import pandas as pd
        
        with tempfile.TemporaryDirectory() as tmpdir:
            service = EventService(
                openagenda_client=Mock(),
                raw_data_path=Path(tmpdir) / 'raw.json',
                processed_data_path=Path(tmpdir) / 'processed.csv'
            )
            
            # Créer un response avec les colonnes requises
            response = {
                'event_1': {
                    'event_id': 'e1',
                    'title.fr': 'Concert',
                    'description.fr': 'Jazz',
                    'location.city': 'Rennes',
                    'location.name': 'Opéra',
                    'location.address': '1 rue',
                    'keywords.fr': 'musique',
                    'dateRange.fr': '15/01/2025',
                    'originAgenda.title': 'Rennes',
                    'firstTiming.begin': '2025-01-15T19:00:00',
                    'firstTiming.end': '2025-01-15T21:00:00',
                    'lastTiming.begin': '2025-01-15T19:00:00',
                    'lastTiming.end': '2025-01-15T21:00:00',
                }
            }
            df = service._transform_to_dataframe(response)
            
            assert isinstance(df, pd.DataFrame)
