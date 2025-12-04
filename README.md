# 📘 README.md — POC RAG “Puls-Events”

**Assistant intelligent de recommandation d’événements culturels (Ille-et-Vilaine)**

## 🎯 Objectif du projet

Le but du POC final sera de construire un système RAG capable de recommander des événements musicaux de la ville de Rennes, en s’appuyant sur :
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
│   ├── config.py          # Variables globales (API keys, etc.)
│   └── __init__.py
│
├── src/                     # Code source principal
│   ├── app/                 # Endpoints API
│   │   ├── main.py          # Point d'entrée FastAPI ou Flask
│   │   ├── routes.py        # Définition des routes
│   │   └── __init__.py
│   │
│   ├── rag/                 # Logique RAG
│   │   ├── retriever.py     # Recherche dans la base de connaissances
│   │   ├── embedder.py      # Génération des embeddings
│   │   ├── pipeline.py      # Orchestration RAG
│   │   └── __init__.py
│   │
│   ├── models/              # Modèles IA (LLM, embeddings)
│   │   ├── mistral_client.py
│   │   └── __init__.py
│   │
│   ├── utils/               # Fonctions utilitaires
│   │   ├── logger.py
│   │   ├── openagenda_client.py
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── data/                    # Données (données,base de connaissances, embeddings) 
│   ├── documents/
│   ├── raw/
│   │  └── events_musique_rennes
│   └── embeddings/
│
├── tests/                   # Tests unitaires et d’intégration
│   ├── test_api.py
│   ├── test_rag.py
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
