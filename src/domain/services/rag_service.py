"""
Service RAG (Retrieval Augmented Generation).
Logique métier pour la recherche et génération de réponses.
"""
from config.logger import logger


class RagService:
    """Service métier pour les opérations RAG"""
    
    def __init__(self, retriever):
        """
        Initialise le service RAG.
        
        Args:
            retriever: Instance du SimpleRetriever (RAG layer)
        """
        self.retriever = retriever
    
    def search_and_answer(self, query: str) -> str:
        """
        Recherche dans l'index et génère une réponse à la question.
        
        Args:
            query: Question de l'utilisateur
            
        Returns:
            Réponse générée par le LLM basée sur le contexte pertinent
            
        Raises:
            Exception: Si l'embedding, la recherche ou la génération échoue
        """
        try:
            logger.info(f"💬 Traitement de la requête: '{query[:50]}...'")
            result = self.retriever.answer_query(query)

            # Certains retrievers retournent un tuple (response, contexts).
            # Normaliser pour toujours renvoyer une chaîne de caractères.
            if isinstance(result, tuple) and len(result) > 0:
                answer = result[0]
            else:
                answer = result

            logger.info("✅ Réponse générée avec succès")
            return answer
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la requête: {e}")
            raise
    
    def search_documents(self, query: str, k: int = 5) -> list[str]:
        """
        Recherche les documents pertinents sans génération de réponse.
        
        Args:
            query: Question ou thème de recherche
            k: Nombre de documents à retourner
            
        Returns:
            Liste des contenus des documents pertinents
            
        Raises:
            Exception: Si l'embedding ou la recherche échoue
        """
        try:
            logger.info(f"🔍 Recherche de documents pour: '{query[:50]}...'")
            documents = self.retriever._embed_query(query)
            logger.info(f"✅ {len(documents)} documents trouvés")
            return documents
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche: {e}")
            raise
