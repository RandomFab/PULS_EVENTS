# -*- coding: utf-8 -*-
"""
Fichier de configuration
Contient les clés API et paramètres globaux.
"""
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
# --- Configuration de l'Application ---
APP_TITLE = "Mix’n’Renn"

# --- Clés API ---
    # --- MISTRAL ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")
    # --- openagenda --- 
OPENAGENDA_API_KEY = os.getenv("OPENAGENDA_API_KEY")

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed"
MODEL_NAME = "mistral-small-latest" # Ou un autre modèle comme mistral-large-latest

# --- Configuration des paramètres d'embeddings ---
CHUNK_SIZE = 800                   # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 150                 # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 64           # Taille des lots pour l'API d'embedding

# --- Configuration de la Recherche ---
SEARCH_K = 5                        # Nombre de documents à récupérer par défaut
RAG_PROMPT = """
Tu es l'assistant culturel de Rennes Métropole, spécialisé dans les événements musicaux.

RÈGLE ABSOLUE :
- Réponds UNIQUEMENT en te basant sur les DOCUMENTS fournis ci-dessous
- Si l'info n'est pas dans les documents, dis-le et propose une alternative

INFORMATIONS À MENTIONNER :
- Nom de l'événement
- Date et horaire
- Lieu
- Style musical (si disponible)

DOCUMENTS :
{context}

QUESTION :
{question}

RÉPONSE (concise, factuelle, basée uniquement sur les documents) :
"""


# --- Configuration des chemins d'acces ---
    # --- racine de base ---
BASE_DIR = Path(__file__).resolve().parent.parent
    # --- openagenda ---
PROCESSED_DATA = BASE_DIR / 'Data' / 'processed' / 'events_rennes_metropole_processed.csv'
RAW_DATA = BASE_DIR / 'Data' / 'raw' / 'events_rennes_metropole_raw.json'
    # --- Embeddings ---
EMBEDDINGS_PATH = BASE_DIR / 'Data' / 'embeddings' / 'embeddings.json'
    # --- Indexation ---
INDEX_PATH = BASE_DIR / "Data" / "index" 
METADATA_PATH = BASE_DIR / "Data" / "index" / 'faiss_metadata.json'
    # --- Evaluation ---
TEST_QUESTION_PATH = BASE_DIR / "src" / "evaluation" / "test_questions.json"
TESTSET_PATH = BASE_DIR / "src" / "evaluation" / "testset.json"