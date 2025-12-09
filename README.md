# 📘 README.md — POC RAG “Puls-Events”

**Assistant intelligent de recommandation d’événements culturels (Ille-et-Vilaine)**

## 🎯 Objectif du projet

Le but du POC final sera de construire un système RAG capable de recommander des événements musicaux de la ville de Rennes sur l'année 2024, en s’appuyant sur :
- l’API OpenAgenda,
- un embedding Mistral,
- un index vectoriel FAISS,
- et une API REST permettant de questionner le chatbot.

## 🧱 Étape 1 — Mise en place de l’environnement 🔧

### 🗂 Structure du projet (prévisionnelle)

L’objectif est d’organiser un dépôt clair et évolutif dès le début.
```
PULS_EVENTS
│
├── README.md                # Documentation du projet
├── requirements.txt         # Dépendances Python
├── .gitignore               # Fichiers à ignorer par Git
├── config/                  # Fichiers de configuration
│   ├── config.py            # Variables globales (API keys, etc.)
│   ├── logger.py            # Logger message infos/warning/error
│   └── __init__.py
│
├── src/                     # Code source principal
│   ├── app/                 # Endpoints API
│   │   ├── main.py          # Point d'entrée FastAPI ou Flask
│   │   ├── routes.py        # Définition des routes
│   │   └── __init__.py
│   │
│   ├── rag/                 # Logique RAG
│   │   ├── chunker.py       # Découpage du texte en chunks
│   │   ├── embedder.py      # Génération des embeddings
│   │   ├── langchain_faiss_indexer.py      # Création des base de données Faiss
│   │   ├── retriever.py     # Recherche dans la base de connaissances + LLM
│   │   └── __init__.py
│   │
│   ├── Scripts/              # Execution d'instance de code
│   │   ├── build_embeddings.py # Création de vecteurs
│   │   ├── build_index_langchain.py # Création de la base faiss
│   │   ├── data_processing.py # récupération et traitement des données openagenda
│   │   └── __init__.py
│   │
│   ├── utils/               # Fonctions utilitaires
│   │   ├── openagenda_client.py # class pour client openagenda
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── data/                    # Données (données,base de connaissances, embeddings) 
│   ├── raw/
│   │  └── events_rennes_metropole_raw.json
│   ├── processed/
│   │  └── events_rennes_metropole_processed.csv
│   ├── embeddings/
│   │  └── embeddings.json
│   └── index
│      └── index.faiss       # Base de données FAISS
├── tests/                   # Tests unitaires et d’intégration
│   ├── test_langchain_search.py # test des chunks retournés 
│   ├── test_retriever_QA.py # test de l'interaction LLM + retriever
│   └── __init__.py
│
└── docker/                  # Fichiers Docker si nécessaire
    ├── Dockerfile
    └── docker-compose.yml
```

### ▶️ Installation et mise en route

1️⃣ **Cloner le dépôt**
```bash
git clone https://github.com/RandomFab/PULS_EVENTS.git
cd PULS_EVENT
```

2️⃣ **Créer un environnement virtuel**
```bash
python -m venv env
source env/bin/activate      # macOS / Linux
env\Scripts\activate         # Windows
```

3️⃣ **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4️⃣ **Tester l’environnement**

Dans un terminal Python / Jupyter :
```python
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from mistral import MistralClient

print("Imports OK 🎉")
```
Si le message s’affiche → environnement validé.

### 📦 requirements.txt 

```
dependencies = [
    "dotenv>=0.9.9",
    "faiss-cpu>=1.13.0",
    "ipykernel>=7.1.0",
    "langchain>=1.1.0",
    "mistralai>=1.9.11",
    "pandas>=2.3.3",
    "requests>=2.32.5",
]
```

### 🔒 La clé API Mistral

La clé API Mistral doit être stockée dans un fichier `.env` non versionné.

### 🌍 Zone géographique ciblée

Le système RAG final ciblera les événements:
- Situés à Rennes (35)
- Musicaux
