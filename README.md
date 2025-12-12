# 📘 PULS_EVENTS - Mix'n'Renn

**Assistant intelligent de recommandation d'événements culturels de Rennes Métropole**

[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-1.1+-orange.svg)](https://python.langchain.com/)
[![Mistral AI](https://img.shields.io/badge/Mistral-AI-purple.svg)](https://mistral.ai/)

## 🎯 Objectif du projet

Système RAG (Retrieval-Augmented Generation) complet pour recommander des événements musicaux de Rennes Métropole en temps réel, utilisant :
- 🌐 **API OpenAgenda** pour les données d'événements
- 🧠 **Mistral AI** pour embeddings et génération de réponses
- 📊 **FAISS** pour l'indexation vectorielle
- ⚡ **FastAPI** pour l'API REST
- 🔗 **LangChain** pour l'orchestration RAG

## ✨ Fonctionnalités

- ✅ **Recherche intelligente** d'événements musicaux par requête naturelle
- ✅ **API REST complète** avec documentation interactive (Swagger)
- ✅ **Scripts CLI** pour automatisation (pipeline complet ou étapes individuelles)
- ✅ **Mise à jour dynamique** des données via endpoints
- ✅ **Évaluation RAG** avec métriques Ragas
- ✅ **Gestion robuste** des erreurs et validation des données
- ✅ **Architecture en couches** professionnelle et évolutive
- ✅ **Services métier réutilisables** (Dependency Inversion Principle)

---


## 📊 Diagrammes Architecturaux

### 🏗️ Diagramme de Composants - Architecture UML

```mermaid
graph TB
    subgraph PRESENTATION["🎨 Presentation Layer"]
        API["FastAPI REST API<br/>main.py<br/>routes.py"]
        CLI["CLI Scripts<br/>run_pipeline.py<br/>process_events.py<br/>generate_embeddings.py"]
    end
    
    subgraph DOMAIN["🧠 Domain Layer - Services Métier"]
        EventSrv["EventService<br/>• fetch_and_process_events<br/>• _validate_dataframe<br/>• _transform_to_dataframe"]
        EmbeddingSrv["EmbeddingService<br/>• build_embeddings<br/>• _chunk_texts<br/>• _save_embeddings"]
        RagSrv["RagService<br/>• search_and_answer<br/>• search_documents<br/>• format_context"]
    end
    
    subgraph RAG["🤖 RAG Layer - Techniques"]
        Embedder["Embedder<br/>• embed_texts<br/>• embed_query"]
        Chunker["Chunker<br/>• chunk_documents<br/>• split_text"]
        Retriever["SimpleRetriever<br/>• answer_query<br/>• search_similar<br/>• _generate_answer"]
        Indexer["FAISS Indexer<br/>• build_from_embeddings<br/>• load_index<br/>• search"]
    end
    
    subgraph INFRA["🔌 Infrastructure Layer"]
        OpenAgenda["OpenAgendaClient<br/>• get_events<br/>• get_events_with_keywords<br/>• _build_params"]
        FileStorage["FileStorage Utils<br/>• save_csv<br/>• save_json<br/>• load_json"]
    end
    
    subgraph EXTERNAL["🌐 Ressources Externes"]
        MistralAPI["Mistral AI API<br/>• Embeddings<br/>• LLM Generation"]
        OpenAgendaAPI["OpenAgenda API<br/>• Events Data<br/>• Keywords"]
    end
    
    subgraph DATA["💾 Data Storage"]
        RawData["Data/raw/<br/>events_rennes.json"]
        ProcessedData["Data/processed/<br/>events_processed.csv"]
        EmbeddingsData["Data/embeddings/<br/>embeddings.json"]
        FaissIndex["Data/index/<br/>faiss_index.index<br/>faiss_metadata.json"]
    end
    
    %% Presentation → Domain
    API --> EventSrv
    API --> EmbeddingSrv
    API --> RagSrv
    CLI --> EventSrv
    CLI --> EmbeddingSrv
    CLI --> RagSrv
    
    %% Domain → RAG + Infrastructure
    EventSrv --> OpenAgenda
    EventSrv --> FileStorage
    EmbeddingSrv --> Embedder
    EmbeddingSrv --> Chunker
    EmbeddingSrv --> FileStorage
    RagSrv --> Retriever
    
    %% RAG → Infrastructure + External
    Retriever --> Embedder
    Retriever --> Indexer
    Embedder --> MistralAPI
    Chunker -->|depends on| Retriever
    
    %% Infrastructure → External
    OpenAgenda -->|calls| OpenAgendaAPI
    Embedder -->|calls| MistralAPI
    
    %% Data flows
    OpenAgenda -->|saves| RawData
    FileStorage -->|saves| ProcessedData
    FileStorage -->|saves| EmbeddingsData
    FileStorage -->|saves| FaissIndex
    
    EventSrv -->|reads/writes| ProcessedData
    EmbeddingSrv -->|reads| ProcessedData
    EmbeddingSrv -->|reads| EmbeddingsData
    Retriever -->|reads| FaissIndex
    Retriever -->|reads| EmbeddingsData
```

### 🔄 Diagramme de Séquence - Requête de Recherche (POST /ask)

```mermaid
sequenceDiagram
    actor User as 👤 Utilisateur
    participant API as 🎨 FastAPI<br/>routes.py
    participant RagSrv as 🧠 RagService
    participant Retriever as 🤖 SimpleRetriever
    participant Embedder as 🔢 Embedder
    participant MistralEmbed as 🟦 Mistral<br/>Embed API
    participant Indexer as 📑 FAISS Indexer
    participant LLM as 🌐 Mistral<br/>LLM API

    User->>API: POST /ask<br/>{"query": "concerts jazz"}
    activate API
    Note over API: 1. Validation Pydantic<br/>2. Initialisation services

    API->>RagSrv: search_and_answer(query)
    activate RagSrv

    RagSrv->>Retriever: answer_query(query)
    activate Retriever

    Note over Retriever: Phase 1: Embedding de la requête
    Retriever->>Embedder: embed_query(query)
    activate Embedder
    Embedder->>MistralEmbed: POST /embeddings<br/>{"input": "concerts jazz"}
    activate MistralEmbed
    MistralEmbed-->>Embedder: embedding vector (1536D)
    deactivate MistralEmbed
    Embedder-->>Retriever: query_vector
    deactivate Embedder

    Note over Retriever: Phase 2: Recherche vectorielle
    Retriever->>Indexer: search_similar(query_vector, k=5)
    activate Indexer
    Note over Indexer: Calcul similarité cosinus<br/>Top-5 documents
    Indexer-->>Retriever: similar_docs_with_scores
    deactivate Indexer

    Note over Retriever: Phase 3: Construction contexte
    Retriever->>Retriever: format_context(documents)

    Note over Retriever: Phase 4: Génération LLM
    Retriever->>LLM: POST /chat.complete<br/>{"messages": [system, user+context]}
    activate LLM
    Note over LLM: Génération réponse<br/>basée sur contexte
    LLM-->>Retriever: Réponse complète
    deactivate LLM

    Retriever-->>RagSrv: answer
    deactivate Retriever

    RagSrv-->>API: answer
    deactivate RagSrv

    API->>API: Format réponse JSON<br/>{"answer": "...", "sources": [...]}

    API-->>User: 200 OK<br/>{"answer": "Les concerts...",<br/>"sources": [events]}
    deactivate API

    Note over User,LLM: ⏱️ Latence: ~2-3 secondes<br/>Dépend de Mistral API
```

### Structure des Fichiers

```
PULS_EVENTS/
│
├── 📚 Documentation
│   ├── README.md                          # Ce fichier
│   └── Documents/
│       ├── COMPONENT_DIAGRAM.md           # Diagramme de composants UML
│       ├── SEQUENCE_DIAGRAM.md            # Diagramme de séquence
│       └── instructions.md                # Instructions du projet
│
├── ⚙️ Configuration
│   ├── config/
│   │   ├── config.py          # Configuration centralisée
│   │   └── logger.py          # Logger configuré
│   ├── .env                   # Variables d'environnement (non versionné)
│   ├── .gitignore
│   ├── requirements.txt
│   └── pyproject.toml
│
├── 🚀 Application
│   └── src/
│       ├── presentation/      # 🎨 Interfaces utilisateur
│       │   ├── api/          # API REST (FastAPI)
│       │   │   ├── main.py
│       │   │   └── routes.py
│       │   └── cli/          # Scripts en ligne de commande
│       │       ├── process_events.py
│       │       ├── generate_embeddings.py
│       │       ├── build_index.py
│       │       └── run_pipeline.py
│       │
│       ├── domain/           # 🧠 Logique métier (cœur)
│       │   ├── models/       # Modèles de données
│       │   └── services/     # Services métier réutilisables
│       │       ├── event_service.py
│       │       ├── embedding_service.py
│       │       ├── evaluation_service.py
│       │       └── rag_service.py
│       │
│       ├── infrastructure/   # 🔌 Accès externe
│       │   └── openagenda_client.py
│       │
│       ├── rag/             # 🤖 Composants RAG techniques
│       │   ├── embedder.py
│       │   ├── chunker.py
│       │   ├── retriever.py
│       │   └── langchain_faiss_indexer.py
│       │
│       ├── evaluation/      # 📊 Évaluation RAG
│       │   ├── build_testset.py
│       │   ├── evaluate_rag.py
│       │   ├── test_questions.json
│       │   └── testset.json
│       │
│       └── utils/           # 🛠️ Utilitaires
│
├── 📊 Données
│   └── Data/
│       ├── raw/             # Données brutes JSON
│       ├── processed/       # Données traitées CSV
│       ├── embeddings/      # Vecteurs d'embeddings
│       ├── index/           # Index FAISS
│       └── documents/       # Documents traités
│
├── 📓 Notebooks
│   └── 01_EDA_JSON_RENNES.ipynb  # Analyse exploratoire des données
│
└── 🧪 Tests
    └── tests/
        ├── test_api.py
        ├── test_chunker.py
        ├── test_cli_scripts.py
        ├── test_embedder.py
        ├── test_embedding_service.py
        ├── test_embedding_service_extended.py
        ├── test_evaluation.py
        ├── test_event_service.py
        ├── test_event_service_extended.py
        ├── test_integration_imports.py
        ├── test_openagenda_client.py
        ├── test_openagenda_extended.py
        ├── test_rag_service.py
        ├── test_retriever_simple.py
        └── test_routes_complete.py
```

---

## 🚀 Installation Rapide

### Prérequis
- Python 3.13+
- pip
- Git

### 1️⃣ Cloner et setup

```bash
# Cloner le dépôt
git clone https://github.com/RandomFab/PULS_EVENTS.git
cd PULS_EVENTS

# Créer l'environnement virtuel
python -m venv .venv

# Activer l'environnement
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Installer les dépendances
pip install -r requirements.txt
```

### 2️⃣ Configuration

Créez un fichier `.env` à la racine :

```env
# Clés API (obligatoires)
MISTRAL_API_KEY=votre_cle_mistral
OPENAGENDA_API_KEY=votre_cle_openagenda
```

### 3️⃣ Initialiser les données

**Option A : Pipeline complet automatique** (recommandé)
```bash
# Exécute toutes les étapes : récupération → embeddings → index
python -m src.presentation.cli.run_pipeline

# Avec dates personnalisées
python -m src.presentation.cli.run_pipeline --start-date 01/06/25 --end-date 31/12/25
```

**Option B : Scripts individuels**
```bash
# 1. Récupérer les événements OpenAgenda
python -m src.presentation.cli.process_events

# 2. Générer les embeddings
python -m src.presentation.cli.generate_embeddings

# 3. Construire l'index FAISS
python -m src.presentation.cli.build_index
```

### 4️⃣ Lancer l'API

```bash
# Mode développement
uvicorn src.presentation.api.main:app --reload

# Mode production
uvicorn src.presentation.api.main:app --host 0.0.0.0 --port 8000
```

### 🐳 Alternative : Utilisation avec Docker Compose

Si vous préférez utiliser Docker, assurez-vous d'avoir Docker et Docker Compose installés.

#### Construire l'image et créer le container
```bash
docker-compose up --build
```

#### Lancer le container (si déjà construit)
```bash
docker-compose up
```

🎉 **L'API est prête !**
- 📖 SWAGGER : http://localhost:8000/docs

---

## 📡 Utilisation de l'API

### Endpoints Principaux

#### 💬 Poser une question
```bash
POST /ask
{
  "query": "Quels sont les concerts rock ce weekend ?"
}
```

#### 🔍 Recherche brute
```bash
POST /search_raw
{
  "query": "jazz"
}
```

#### 🔄 Mettre à jour les données
```bash
POST /update_datas
{
  "start_date": "01/01/25",
  "end_date": "31/12/25"
}
```

#### 🏗️ Reconstruire l'index
```bash
POST /rebuild
```

#### 📊 Évaluer le RAG
```bash
GET /evaluate_rag
```

---

## 🧪 Tests

```bash
# Tous les tests
pytest

# Tests spécifiques
pytest tests/test_api.py

# Avec couverture
pytest --cov=src
```

---

## 🔒 Sécurité

⚠️ **Ne jamais commiter le fichier `.env`**  
Les clés API doivent rester secrètes.

---

## 🐛 Dépannage

### "Module not found"
```bash
pip install -r requirements.txt
```

### "MISTRAL_API_KEY not found"
Vérifier que `.env` existe et contient la clé.

### "Index FAISS introuvable"
```bash
python -m src.presentation.cli.build_index
# Ou exécuter le pipeline complet
python -m src.presentation.cli.run_pipeline
```

### "ModuleNotFoundError: No module named 'src.app'"
L'ancienne architecture a été refactorisée. Utilisez les nouveaux chemins :
- ❌ `src.app.main` → ✅ `src.presentation.api.main`


---

## 👤 Auteur

**Fabien** - [RandomFab](https://github.com/RandomFab)

---

## 🙏 Remerciements

- [Mistral AI](https://mistral.ai/) pour les modèles
- [OpenAgenda](https://openagenda.com/) pour l'API
- [LangChain](https://python.langchain.com/) pour le framework
- [FastAPI](https://fastapi.tiangolo.com/) pour l'API

---

**⭐ Si ce projet vous est utile, donnez-lui une étoile !**
```
