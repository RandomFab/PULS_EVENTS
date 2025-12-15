"""
Tests pour les CLI scripts.
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner


class TestCLIScripts:
    """Tests basiques des scripts CLI"""
    
    def test_cli_modules_are_importable(self):
        """Vérifie que les modules CLI sont importables"""
        from src.presentation.cli import build_index
        from src.presentation.cli import generate_embeddings
        from src.presentation.cli import process_events
        
        assert build_index is not None
        assert generate_embeddings is not None
        assert process_events is not None
    
    def test_run_pipeline_module_exists(self):
        """Vérifie que le module run_pipeline existe"""
        from src.presentation.cli import run_pipeline
        assert run_pipeline is not None
