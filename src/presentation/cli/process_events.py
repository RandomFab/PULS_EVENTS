"""
Script CLI pour récupérer et traiter les données OpenAgenda.
Point d'entrée en ligne de commande pour la mise à jour des événements.
"""
from config.config import OPENAGENDA_API_KEY, RAW_DATA, PROCESSED_DATA
from config.logger import logger
from src.infrastructure.openagenda_client import OpenAgendaClient
from src.domain.services.event_service import EventService

if __name__ == "__main__":
    logger.info("🚀 Démarrage de la récupération des données OpenAgenda...")
    
    # Initialisation de l'infrastructure
    openagenda_client = OpenAgendaClient(api_key=OPENAGENDA_API_KEY)
    
    # Initialisation du service
    event_service = EventService(
        openagenda_client=openagenda_client,
        raw_data_path=RAW_DATA,
        processed_data_path=PROCESSED_DATA
    )
    
    # Exécution : dates par défaut (2025-2027)
    event_service.fetch_and_process_events(
        start_date='01/01/25',
        end_date='01/01/27'
    )
    
    logger.info("✅ Récupération et traitement des données terminés !")
