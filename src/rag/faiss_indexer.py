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
            logger.error("❌ Fichier non trouvé → vérifier le chemin d'accès.")
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
    
    def load_index(self):

        try:
            logger.info("🔄 Chargement des index...")
            self.index = faiss.read_index(str(self.index_path))
            logger.info("✅ Index chargés avec succès !")
        except FileNotFoundError :
            logger.error("❌ Fichier non trouvé → vérifier le chemin d'accès.")
            self.index=[]
        except Exception as e :
            logger.error(f"❌ Echec lors du chargement des index → {e}")
            self.index=[]


        try:
            logger.info("🔄 Chargement des metadatas...")
            with open(self.metadata_path,'r',encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info("✅ metadatas chargés avec succès !")
        
        except FileNotFoundError :
            logger.error("❌ Fichier non trouvé → vérifier le chemin d'accès.")
            self.metadata=[]
        except json.JSONDecodeError :
            logger.error("❌ Le fichier json n'a pas pu être décodé.")
            self.metadata=[]
        except Exception as e :
            logger.error(f"❌ Echec lors du chargement des metadats → {e}")
            self.metadata=[]

    def search(self,query_vector:np.ndarray,k:int=5):

        if (self.index is None) or (self.metadata == []):
            raise ValueError("""❌ Index ou metadatas non chargés ! 
                             Essayez de lancer un FaissIndexer.load_index() ou 
                             FaissIndexer.index_embeddigns()""")
        
        query_vector = np.array(query_vector).astype('float32')
        
        logger.info(f"Shape du query_vector : {query_vector.shape}")

        distances, indices = self.index.search(query_vector,k)

        results=[]
        for i in range(len(query_vector)):
            query_result = []
            for dist, idx in zip(distances[i],indices[i]):
                if idx == -1:
                    continue

                logger.info(f"index : {idx}, Distance  : {dist}")
                
                meta = self.metadata[idx].copy()
                meta['distance'] = float(dist)

                query_result.append(meta)
            results.append(query_result)
        
        return results
        
