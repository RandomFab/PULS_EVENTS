"""
Tests pour Chunker.
"""
import pytest
from src.rag.chunker import Chunker


class TestChunkerInit:
    """Tests d'initialisation"""
    
    def test_chunker_default_init(self):
        """Vérifie l'initialisation avec les paramètres par défaut"""
        chunker = Chunker()
        assert chunker.splitter is not None
    
    def test_chunker_custom_init(self):
        """Vérifie l'initialisation avec paramètres personnalisés"""
        chunker = Chunker(chunk_size=1000, chunk_overlap=200)
        assert chunker.splitter is not None


class TestSplitText:
    """Tests de la méthode split_text"""
    
    def test_split_basic_text(self):
        """Vérifie la division d'un texte de base"""
        chunker = Chunker()
        text = "Contenu " * 100
        
        splits = chunker.split_text(text)
        
        assert isinstance(splits, list)
        assert len(splits) > 0
        assert all(isinstance(s, str) for s in splits)
    
    def test_split_empty_string(self):
        """Vérifie le comportement avec une chaîne vide"""
        chunker = Chunker()
        splits = chunker.split_text("")
        
        assert isinstance(splits, list)
        assert len(splits) == 0
    
    def test_split_short_text(self):
        """Vérifie avec un texte court"""
        chunker = Chunker()
        splits = chunker.split_text("Texte court")
        
        assert isinstance(splits, list)
        assert len(splits) >= 1
    
    def test_split_long_text(self):
        """Vérifie avec un texte long"""
        chunker = Chunker()
        text = "Ligne " * 5000
        
        splits = chunker.split_text(text)
        
        assert len(splits) > 1
    
    def test_split_preserves_content(self):
        """Vérifie que le contenu complet est préservé"""
        chunker = Chunker()
        text = "Concert de jazz. Festival rock. Ballet classique. " * 50
        
        splits = chunker.split_text(text)
        joined = " ".join(splits)
        
        # Le contenu original doit être présent (avec chevauchements)
        assert "Concert" in joined
        assert "jazz" in joined


class TestChunkerEdgeCases:
    """Tests des cas limites"""
    
    def test_special_characters(self):
        """Vérifie avec des caractères spéciaux"""
        chunker = Chunker()
        text = "Événement spécial: (musique) [festival] {danse} - 2025!"
        
        splits = chunker.split_text(text)
        
        assert len(splits) >= 1
        assert all(isinstance(s, str) for s in splits)
    
    def test_unicode_content(self):
        """Vérifie avec du contenu Unicode"""
        chunker = Chunker()
        text = "Événement à Rennes: café, crêpes, été, été... " * 100
        
        splits = chunker.split_text(text)
        
        assert len(splits) > 0
        assert all(isinstance(s, str) for s in splits)
    
    def test_newlines_and_spaces(self):
        """Vérifie avec plusieurs sauts de ligne"""
        chunker = Chunker()
        text = "Ligne 1\n\nLigne 2\n\nLigne 3\n\n" * 100
        
        splits = chunker.split_text(text)
        
        assert len(splits) > 0
