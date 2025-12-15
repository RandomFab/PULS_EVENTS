"""
Service de gestion des événements.
Logique métier pour récupérer, traiter et valider les événements.
"""
from pathlib import Path
import pandas as pd
import json
from typing import Optional
from config.logger import logger


class EventService:
    """Service métier pour la gestion des événements"""
    
    # Colonnes requises dans les données OpenAgenda
    REQUIRED_COLUMNS = [
        'event_id', 'keywords.fr', 'dateRange.fr',
        'description.fr', 'originAgenda.title', 'title.fr',
        'lastTiming.end', 'lastTiming.begin', 'firstTiming.end',
        'firstTiming.begin', 'location.address',
        'location.name', 'location.city'
    ]
    
    def __init__(self, openagenda_client, raw_data_path: Path, processed_data_path: Path):
        """
        Initialise le service d'événements.
        
        Args:
            openagenda_client: Client pour l'API OpenAgenda
            raw_data_path: Chemin pour sauvegarder les données brutes
            processed_data_path: Chemin pour sauvegarder les données traitées
        """
        self.client = openagenda_client
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
    
    def fetch_and_process_events(
        self, 
        start_date: str = '01/01/25',
        end_date: str = '01/01/27',
        agenda_uid: str = '20500020',
        keywords: Optional[list[str]] = None
    ) -> pd.DataFrame:
        """
        Récupère et traite les événements depuis l'API OpenAgenda.
        
        Args:
            start_date: Date de début au format 'DD/MM/YY'
            end_date: Date de fin au format 'DD/MM/YY'
            agenda_uid: Identifiant de l'agenda OpenAgenda
            keywords: Liste de mots-clés pour filtrer (défaut: ['musique', 'concert'])
        
        Returns:
            DataFrame des événements traités
            
        Raises:
            ValueError: Si les colonnes requises sont manquantes
        """
        if keywords is None:
            keywords = ['musique', 'concert']
        
        logger.info(f"🔄 Récupération des événements du {start_date} au {end_date}...")
        
        # Récupération des données brutes
        response = self.client.get_events_with_keywords(
            agendaUID=agenda_uid,
            start_date=start_date,
            end_date=end_date,
            keywords=keywords,
            params={'limit': 1000}
        )
        
        # Sauvegarde des données brutes
        self._save_raw_data(response)
        logger.info(f"✅ Données brutes sauvegardées : {len(response)} événements")
        
        # Transformation et validation
        df = self._transform_to_dataframe(response)
        self._validate_dataframe(df)
        
        # Sauvegarde des données traitées
        self._save_processed_data(df)
        logger.info(f"✅ Données traitées sauvegardées : {len(df)} événements avec {len(self.REQUIRED_COLUMNS)} colonnes")
        
        return df
    
    def load_processed_events(self) -> pd.DataFrame:
        """
        Charge les événements traités depuis le fichier CSV.
        
        Returns:
            DataFrame des événements
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        if not self.processed_data_path.exists():
            raise FileNotFoundError(
                f"Fichier de données traité introuvable : {self.processed_data_path}. "
                "Exécutez d'abord le traitement des données."
            )
        
        return pd.read_csv(self.processed_data_path)
    
    def _save_raw_data(self, response: dict) -> None:
        """Sauvegarde les données brutes JSON"""
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.raw_data_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=4, ensure_ascii=False)
    
    def _transform_to_dataframe(self, response: dict) -> pd.DataFrame:
        """Transforme la réponse JSON en DataFrame"""
        events_list = [
            {"event_id": key, **value} for key, value in response.items()
        ]
        df = pd.json_normalize(events_list)
        return df[self.REQUIRED_COLUMNS]
    
    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Valide que le DataFrame contient toutes les colonnes requises.
        
        Raises:
            ValueError: Si des colonnes sont manquantes
        """
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ Colonnes manquantes : {missing_columns}")
            raise ValueError(
                f"Colonnes manquantes : {missing_columns}. "
                "L'API OpenAgenda a peut-être changé de format."
            )
    
    def _save_processed_data(self, df: pd.DataFrame) -> None:
        """Sauvegarde le DataFrame traité en CSV"""
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_data_path, index=False)
