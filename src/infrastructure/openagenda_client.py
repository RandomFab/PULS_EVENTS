"""
Client pour l'API OpenAgenda.
Infrastructure - Accès externe aux données événementielles.
"""
import requests
from datetime import datetime
from config.logger import logger


class OpenAgendaClient:
    """Client HTTP pour interagir avec l'API OpenAgenda"""
    
    BASE_URL = 'https://api.openagenda.com/v2/agendas'

    def __init__(self, api_key: str):
        """
        Initialise le client OpenAgenda.
        
        Args:
            api_key: Clé API pour l'authentification
        """
        self.api_key = api_key

    def get_events(
        self, 
        agendaUID: str,
        start_date: str = None,
        end_date: str = None,
        params: dict = None
    ) -> dict:
        """
        Récupère les événements d'un agenda OpenAgenda entre deux dates (incluses).
        
        La méthode :
        - Construit une requête GET vers '{BASE_URL}/{agendaUID}/events'
        - Convertit les dates fournies au format ISO 8601 UTC (YYYY-MM-DDTHH:MM:SSZ)
        - Injecte ces valeurs dans les paramètres de requête sous les clés 'timings[gte]' et 'timings[lte]'
        - Retourne la réponse au format JSON
        
        Args:
            agendaUID: Identifiant (UID) de l'agenda OpenAgenda à interroger.
                Exemple : "20500020" (Rennes métropole)
            start_date: Date de début (incluse) au format "DD/MM/YY".
                Exemple : "01/01/24"
            end_date: Date de fin (incluse) au format "DD/MM/YY".
                Exemple : "01/01/27"
            params: Dictionnaire des paramètres de requête supplémentaires à transmettre à l'API.
                Exemples de paramètres supportés :
                - "q": Filtre par mot-clé (titre, description, tags)
                - "limit": Limite le nombre de résultats retournés (par page)
                - "offset": Décalage pour la pagination (commence à 0)
                - "keyword": Recherche par mot-clé spécifique
                - "adminLevel2[]": Filtre par département (ex: Ille-et-Vilaine)
                - "adminLevel3[]": Filtre par commune (ex: Rennes)
                - "tags[]": Filtre par tag (ex: Concert, Festival)
        
        Returns:
            Le corps JSON décodé renvoyé par l'API OpenAgenda
            
        Raises:
            requests.HTTPError: Si la requête échoue (4xx, 5xx)
        """
        url = f'{self.BASE_URL}/{agendaUID}/events'
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Conversion en format ISO 8601 UTC
        start_date_UTC = datetime.strptime(start_date, "%d/%m/%y").strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date_UTC = datetime.strptime(end_date, "%d/%m/%y").strftime("%Y-%m-%dT%H:%M:%SZ")

        if params is None:
            params = {}
        
        if start_date:
            params['timings[gte]'] = start_date_UTC
        if end_date:
            params['timings[lte]'] = end_date_UTC

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_events_with_keywords(
        self,
        keywords: list[str],
        agendaUID: str,
        start_date: str = None,
        end_date: str = None,
        params: dict = None
    ) -> dict:
        """
        Récupère les événements filtrés par mots-clés avec gestion des variations.
        
        Pour chaque mot-clé fourni, cette méthode :
        1. Génère les variations (minuscule, majuscule, capitalisée)
        2. Effectue une recherche pour chaque variation
        3. Déduplique les résultats par UID
        
        Args:
            keywords: Liste de mots-clés pour filtrer (ex: ['musique', 'concert'])
            agendaUID: Identifiant de l'agenda OpenAgenda
            start_date: Date de début au format "DD/MM/YY"
            end_date: Date de fin au format "DD/MM/YY"
            params: Paramètres supplémentaires pour l'API
        
        Returns:
            Dictionnaire {uid: event} avec tous les événements trouvés (dédupliqués)
        """
        all_events = {}

        if params is None:
            params = {}

        for keyword in keywords:
            # Génération des variations de casse
            variations = {keyword, keyword.lower(), keyword.upper(), keyword.capitalize()}

            for v in variations:
                try:
                    reset_params = params.copy()
                    reset_params['keyword'] = v

                    data = self.get_events(
                        agendaUID=agendaUID,
                        start_date=start_date,
                        end_date=end_date,
                        params=reset_params
                    )
                    events = data.get('events', [])
                    
                    # Déduplication par UID
                    for ev in events:
                        all_events[ev['uid']] = ev
                        
                except Exception as e:
                    # Continue avec les autres variations même si une échoue
                    logger.warning(f"⚠️ Erreur pour la variation '{v}': {e}")
                    continue

        return all_events
