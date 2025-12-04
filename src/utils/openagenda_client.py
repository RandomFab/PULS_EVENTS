import requests
from config.config import OPENAGENDA_API_KEY
import json
from datetime import datetime

class OpenAgendaClient:
    BASE_URL = 'https://api.openagenda.com/v2/agendas'

    def __init__(self, api_key):
        self.api_key  = api_key


    def get_events(self, agendaUID: str,start_date:str=None,end_date:str=None, params=None):
        '''
            Récupère les événements d'un agenda OpenAgenda entre deux dates (incluses).
            La méthode :
                - Construit une requête GET vers '{BASE_URL}/{agendaUID}/events',
                - Convertit les dates fournies au format ISO 8601 UTC(YYYY-MM-DDTHH:MM:SSZ)
                - Injecte ces valeurs dans les paramètres de requête sous les clés 'timings[gte]' et 'timings[lte]'
                - Retourne la réponse au format JSON.
            ARGS :
                agendaUID (str)
                    Identifiant (UID) de l'agenda OpenAgenda à interroger.
                    Exemple : "20500020" (Rennes métropole)
                start_date (str)
                    Date de début (incluse) -> "DD/MM/YY".
                    Exemple : "01/01/24" 
                end_date (str)
                    Date de fin (incluse) -> "DD/MM/YY".
                    Exemple : "01/01/24" 
                params (dict, optional)
                    Dictionnaire des paramètres de requête supplémentaires à transmettre à l'API.
                    Exemple : 
                        params = {
                            "q": "festival",  # Filtre par mot-clé (titre, description, tags)
                            "limit": 20,  # Limite le nombre de résultats retournés (par page)
                            "offset": 0,  # Décalage pour la pagination (commence à 0)
                            
                            # --- Filtres géographiques ---
                            "adminLevel2[]": "Ille-et-Vilaine",  # Filtre par département (ex: Ille-et-Vilaine)
                            "adminLevel3[]": "Rennes",  # Filtre par commune (ex: Rennes)
                            "location[]": "Rennes",  # Filtre par lieu précis (nom du lieu)
                            "postalCode[]": "35000",  # Filtre par code postal
                            
                            # --- Filtres temporels ---
                            "timings[gte]": "2024-06-01T00:00:00Z",  # Date de début (>=) en format ISO 8601 UTC
                            "timings[lte]": "2024-08-31T23:59:59Z",  # Date de fin (<=) en format ISO 8601 UTC
                            
                            # --- Filtres par type ou catégorie ---
                            "tags[]": "Concert",  # Filtre par tag (ex: Concert, Festival)
                            "category[]": "Musique",  # Filtre par catégorie (si disponible dans l’agenda)
                            
                            # --- Filtres par accessibilité ---
                            "accessibility[]": "wheelchair",  # Filtre par accessibilité (ex: fauteuil roulant)
                            
                            # --- Filtres par prix ---
                            "price[gte]": 0,  # Prix minimum (>=)
                            "price[lte]": 50,  # Prix maximum (<=)
                            
                            # --- Filtres par statut ---
                            "status[]": "confirmed",  # Statut de l’événement (ex: confirmé, annulé)
                            
                            # --- Filtres par organisateur ---
                            "organizer[]": "Ville de Rennes",  # Filtre par nom d’organisateur
                            
                            # --- Filtres par langue ---
                            "lang[]": "fr",  # Filtre par langue (ex: fr, en)
                        }
            RETURN :
                dict
                    Le corps JSON décodé renvoyé par l'API OpenAgenda (format dépendant de l'API).
                    Exemple de structure de retour typique :
        '''

        url = f'{self.BASE_URL}/{agendaUID}/events'
        headers = {"Authorization": f"Bearer {self.api_key}"}

        # Conversion en objet datetime + Conversion en format ISO 8601 UTC
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
    
    def get_events_with_keywords(self,
                                keywords:list[str],
                                agendaUID: str,
                                start_date:str=None,
                                end_date:str=None, 
                                params=None):
        all_events= {}

        if params is None:
            params={}

        for keyword in keywords:
            
            variations = {keyword, keyword.lower(),keyword.upper,keyword.capitalize()}

            for v in variations:
                
                reset_params = params.copy()

                reset_params['keyword'] = v

                data = self.get_events(agendaUID=agendaUID,
                                start_date=start_date,
                                end_date=end_date,
                                params=reset_params)
                events = data['events']
                
                for ev in events:
                    all_events[ev['uid']] = ev

        return all_events
        






if __name__ == '__main__':

    client = OpenAgendaClient(api_key=OPENAGENDA_API_KEY)
    keywords = ['musique','concert']
    response = client.get_events_with_keywords(agendaUID='20500020',
                                               start_date='01/01/24',
                                               end_date='01/01/25',
                                               keywords=keywords,
                                               params = {'limit':1000,
                                                        'city[]':['Rennes','rennes']}
                                            )

    with open('Data/raw/events_rennes_metropole.json','w',encoding='utf-8') as f:
        json.dump(response, f, indent=4, ensure_ascii=False)


