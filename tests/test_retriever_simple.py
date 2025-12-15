"""
Tests pour SimpleRetriever.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.rag.retriever import SimpleRetriever


class TestSimpleRetrieverInit:
    """Tests d'initialisation du SimpleRetriever"""
    
    @patch('src.rag.retriever.Embedder')
    @patch('src.rag.retriever.ChatMistralAI')
    @patch('src.rag.retriever.LangChainFaissIndexer')
    def test_init_creates_embedder(self, mock_indexer, mock_llm, mock_embedder):
        """Vérifie la création de l'embedder"""
        retriever = SimpleRetriever(
            index_dir='test_index',
            api_key='test-key'
        )
        
        assert mock_embedder.called
    
    @patch('src.rag.retriever.Embedder')
    @patch('src.rag.retriever.ChatMistralAI')
    @patch('src.rag.retriever.LangChainFaissIndexer')
    def test_init_creates_llm(self, mock_indexer, mock_llm, mock_embedder):
        """Vérifie la création du LLM"""
        retriever = SimpleRetriever(
            index_dir='test_index',
            api_key='test-key'
        )
        
        assert mock_llm.called
    
    @patch('src.rag.retriever.Embedder')
    @patch('src.rag.retriever.ChatMistralAI')
    @patch('src.rag.retriever.LangChainFaissIndexer')
    def test_init_creates_indexer(self, mock_indexer, mock_llm, mock_embedder):
        """Vérifie la création de l'indexer"""
        retriever = SimpleRetriever(
            index_dir='test_index',
            api_key='test-key'
        )
        
        assert mock_indexer.called


class TestSimpleRetrieverMethods:
    """Tests des méthodes du SimpleRetriever"""
    
    @patch('src.rag.retriever.Embedder')
    @patch('src.rag.retriever.ChatMistralAI')
    @patch('src.rag.retriever.LangChainFaissIndexer')
    def test_embed_query_method_exists(self, mock_indexer, mock_llm, mock_embedder):
        """Vérifie que la méthode _embed_query existe"""
        retriever = SimpleRetriever(
            index_dir='test_index',
            api_key='test-key'
        )
        
        assert hasattr(retriever, '_embed_query')
    
    @patch('src.rag.retriever.Embedder')
    @patch('src.rag.retriever.ChatMistralAI')
    @patch('src.rag.retriever.LangChainFaissIndexer')
    def test_answer_query_method_exists(self, mock_indexer, mock_llm, mock_embedder):
        """Vérifie que la méthode answer_query existe"""
        retriever = SimpleRetriever(
            index_dir='test_index',
            api_key='test-key'
        )
        
        assert hasattr(retriever, 'answer_query')
