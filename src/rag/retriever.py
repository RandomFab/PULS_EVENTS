from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate 
from langchain_mistralai import ChatMistralAI
from src.rag.langchain_faiss_indexer import LangChainFaissIndexer
from src.rag.embedder import Embedder
from langchain_core.output_parsers import StrOutputParser
from config.config import APP_TITLE,MODEL_NAME,MISTRAL_API_KEY,EMBEDDING_MODEL,RAG_PROMPT
from config.logger import logger

class SimpleRetriever():
    def __init__(self,index_dir:str,api_key:str,embedding_model:str='mistral-embed',model_name:str = "mistral-small-latest", embeddings_path:str = None):
        """
        Initialise le retriever avec les composants nécessaires.
        
        Args:
            index_dir: Répertoire contenant l'index FAISS
            api_key: Clé API Mistral
            embedding_model: Modèle d'embedding à utiliser
            model_name: Modèle de chat à utiliser
            embeddings_path: Chemin vers le fichier embeddings.json (optionnel)
        """
        try:
            logger.info("🚀 Initialisation du SimpleRetriever...")
            
            # Initialisation de l'embedder
            logger.debug(f"📊 Initialisation de l'embedder avec {embedding_model}")
            self.embedder = Embedder(api_key=api_key, model=embedding_model)
            
            # Initialisation du LLM avec paramètres optimisés pour chatbot culturel
            logger.debug(f"🤖 Initialisation du LLM avec {model_name}")
            self.llm = ChatMistralAI(
                api_key=api_key, 
                model_name=model_name,
                temperature=0.5,    # Augmenté pour plus de créativité dans les recommandations
                top_p=0.9,
                max_tokens=1024     # Doublé pour permettre des réponses plus complètes
            )
            
            # Initialisation de l'indexer
            logger.debug(f"📚 Initialisation de l'indexer avec {index_dir}")
            self.indexer = LangChainFaissIndexer(index_dir=index_dir, embedding_client=self.embedder.client)
            self.indexer.embeddings_path = embeddings_path or "Data/embeddings/embeddings.json"
            
            logger.info("✅ SimpleRetriever initialisé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du SimpleRetriever: {e}")
            raise
    
    def _embed_query(self,query:str):
        """
        Embed la requête et effectue la recherche dans l'index.
        
        Args:
            query: La requête de recherche
            
        Returns:
            Liste des contenus des documents pertinents
        """
        try:
            logger.info(f"🔍 Traitement de la requête: '{query[:50]}...'")
            
            # Embedding de la requête
            logger.debug("🧠 Génération de l'embedding pour la requête...")
            query_vector = self.embedder.embed_texts([query])[0]
            logger.debug(f"✅ Embedding généré (dimension: {len(query_vector)})")
            
            # Chargement de l'index
            logger.debug("📂 Chargement de l'index FAISS...")
            self.indexer.load()
            
            # Recherche par vecteur
            logger.debug("🔎 Recherche dans l'index...")
            documents = self.indexer.search(query=query, k=10)
            
            # Extraction du contenu
            retriever = [doc.page_content for doc, score in documents]
            logger.info(f"✅ {len(retriever)} documents pertinents trouvés")
            
            return retriever
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'embedding/recherche de la requête: {e}")
            raise

    def answer_query(self, query:str ):
        """
        Répond à une requête en utilisant le contexte récupéré.
        
        Args:
            query: La question de l'utilisateur
            
        Returns:
            La réponse générée par le LLM
        """
        try:
            logger.info(f"💬 Génération de la réponse pour: '{query[:50]}...'")
            
            # Récupération du contexte
            logger.debug("📖 Récupération du contexte pertinent...")
            retriever = self._embed_query(query=query)
            
            if not retriever:
                logger.warning("⚠️ Aucun contexte trouvé pour la requête")
                return "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question."
            
            # Construction du template
            template = ChatPromptTemplate.from_template(RAG_PROMPT)
            
            # Construction de la chaîne (simplifié)
            logger.debug("🔗 Construction de la chaîne de génération...")
            chain = template | self.llm | StrOutputParser()
            
            # Génération de la réponse
            logger.debug("✨ Génération de la réponse...")
            context_text = "\n".join(retriever)
            response = chain.invoke({
                "context": context_text,
                "question": query
            })
            
            logger.info("✅ Réponse générée avec succès")
            return response, retriever
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de la réponse: {e}")
            return "Désolé, une erreur s'est produite lors du traitement de votre requête. Veuillez réessayer.",[]