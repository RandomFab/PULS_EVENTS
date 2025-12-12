"""
Integration tests for coverage improvement.
"""
import pytest
from pathlib import Path
import tempfile
import json


class TestModuleImports:
    """Test that all modules can be imported successfully"""
    
    def test_all_cli_modules_import(self):
        """Vérifie que tous les modules CLI s'importent"""
        from src.presentation.cli import build_index
        from src.presentation.cli import generate_embeddings
        from src.presentation.cli import process_events
        from src.presentation.cli import run_pipeline
        
        assert all([build_index, generate_embeddings, process_events, run_pipeline])
    
    def test_all_evaluation_modules_import(self):
        """Vérifie que tous les modules d'évaluation s'importent"""
        from src.evaluation import build_testset
        from src.evaluation import evaluate_rag
        
        assert all([build_testset, evaluate_rag])
    
    def test_all_rag_modules_import(self):
        """Vérifie que tous les modules RAG s'importent"""
        from src.rag import chunker
        from src.rag import embedder
        from src.rag import retriever
        from src.rag import langchain_faiss_indexer
        
        assert all([chunker, embedder, retriever, langchain_faiss_indexer])
    
    def test_all_domain_modules_import(self):
        """Vérifie que tous les modules domain s'importent"""
        from src.domain.services import embedding_service
        from src.domain.services import event_service
        from src.domain.services import rag_service
        from src.domain.services import evaluation_service
        
        assert all([embedding_service, event_service, rag_service, evaluation_service])
    
    def test_api_modules_import(self):
        """Vérifie l'import des modules API"""
        from src.presentation.api import main
        from src.presentation.api import routes
        
        assert all([main, routes])


class TestConfigModules:
    """Test configuration modules"""
    
    def test_config_can_import(self):
        """Vérifie que la configuration s'importe"""
        from config import config
        from config import logger
        
        assert config is not None
        assert logger is not None


class TestUtilityModules:
    """Test utility modules"""
    
    def test_utils_import(self):
        """Vérifie l'import des utils"""
        from src import utils
        
        assert utils is not None
