import os
import json
import faiss
import numpy as np
import logging
from tqdm import tqdm
from config.logger import logger

class FaissIndexer:

    def __init__(self,index_path:str,metadata_path:str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []

    def save_index(self):

        logger.info("💾 Sauvegarde de l'index Faiss sur le disque...")
        try :
            faiss.write_index(self.index,str(self.index_path))
            logger.info("✅ L'index Faiss a été sauvegardé !")
        except Exception as e:
            logger.error(f"❌ Echec lors de la sauvergarde des index → {e}")

        logger.info("💾 Sauvegarde des metadatas sur le disque...")
        try :
            with open(self.metadata_path,'w',encoding='utf-8') as f:
                json.dump(self.metadata, f,ensure_ascii=False,indent=2)
            logger.info("✅ Les metadatas ont été sauvegardées !")
        except Exception as e:
            logger.error(f"❌ Echec lors de la sauvegarde des metadatas → {e}")


    def index_embeddings(self, embeddings_path:str):
        """
        Construit l'index FAISS à partir d'un fichier JSON d'embeddings.
        """

        try : 
            logger.info("🔄 Chargement de la base de données embeddings")

            with open(embeddings_path,'r',encoding='utf-8') as f:
                embeddings = json.load(f)
            logger.info("✅ Embeddings chargés avec succès !")
        except FileNotFoundError :
            logger.error("❌ Fichier non trouver → vérifier le chemin d'accès.")
            embeddings=[]
        except json.JSONDecodeError :
            logger.error("❌ Le fichier json n'a pas pu être décodé.")
            embeddings=[]
        except Exception as e :
            logger.error(f"❌ Echec lors du chargement des embeddings → {e}")
            embeddings=[]
        
        vectors = []
        metadata = []

        for item in tqdm(embeddings):
            vectors.append(item["vector"])
            metadata.append({
                "event_id": item["event_id"],
                "chunk": item["chunk"],
                "title": item["title"],
                "keywords": item["keywords"],
                "start_date": item["start_date"],
                "end_date": item["end_date"],
                "location_address": item["location_adress"],
                "location_name": item["location_name"]
            })

        vectors = np.array(vectors).astype("float32")
        dimension = vectors.shape[1]

        logger.info(f"📝 Création de la base Faiss (dimension = {dimension}) en cours ...")

        try: 

            index = faiss.IndexFlatL2(dimension)
            index.add(vectors)
            logger.info(f"✅ La base Faiss a été créée et  {index.ntotal} vecteurs ont été indexés !")

        except Exception as e:
            logger.error(f"❌ Echec lors de la création de la base Faiss → {e}")
            index = None

        
        self.index = index
        self.metadata = metadata

        

        