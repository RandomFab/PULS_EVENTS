from src.utils.openagenda_client import OpenAgendaClient
from config.config import OPENAGENDA_API_KEY
import pandas as pd
import json

client = OpenAgendaClient(api_key=OPENAGENDA_API_KEY)
keywords = ['musique','concert']
response = client.get_events_with_keywords(agendaUID='20500020',
                                            start_date='01/01/24',
                                            end_date='01/01/25',
                                            keywords=keywords,
                                            params = {'limit':1000,
                                                    }
                                        )

with open('Data/raw/events_rennes_metropole.json','w',encoding='utf-8') as f:
    json.dump(response, f, indent=4, ensure_ascii=False)

events_list = [
    {"event_id": key, **value} for key, value in response.items()
]
df = pd.json_normalize(events_list)
columns_to_keep = ['event_id','keywords.fr','dateRange.fr',
                    'description.fr','originAgenda.title','title.fr',
                    'lastTiming.end','lastTiming.begin','firstTiming.end',
                    'firstTiming.begin','location.address',
                    'location.name','location.city']
df= df[columns_to_keep]
df.to_csv(".../Data/processed/events_musique_35.csv")