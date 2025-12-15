"""
Tests pour les modules d'évaluation.
"""
import pytest
from unittest.mock import Mock, patch
import json
import tempfile
from pathlib import Path


class TestEvaluationModules:
    """Tests basiques des modules d'évaluation"""
    
    def test_build_testset_module_exists(self):
        """Vérifie que le module build_testset existe"""
        from src.evaluation import build_testset
        assert build_testset is not None
    
    def test_evaluate_rag_module_exists(self):
        """Vérifie que le module evaluate_rag existe"""
        from src.evaluation import evaluate_rag
        assert evaluate_rag is not None


class TestEvaluationService:
    """Tests du service d'évaluation"""
    
    def test_evaluation_service_module_exists(self):
        """Vérifie que le module evaluation_service existe"""
        from src.domain.services import evaluation_service
        assert evaluation_service is not None
