# Rapport Technique — Assistant de recommandation d'événements culturels (PULS_EVENTS)

Ce document suit la structure fournie et décrit le POC développé dans le dépôt PULS_EVENTS.

---

1. Objectifs du projet

- Contexte : Présentez la mission confiée par Puls-Events.

  La mission confiée par Puls-Events consiste à prototyper un assistant intelligent capable de recommander des événements culturels (principalement concerts) sur le territoire de Rennes Métropole. Le système doit alimenter des réponses en langage naturel basées sur des données d'événements extraites via l'API OpenAgenda et enrichies par des embeddings pour permettre une recherche sémantique (RAG).

- Problématique : En quoi un système RAG répond-il aux besoins métier ?

  Les besoins métier incluent : fournir des recommandations pertinentes à partir d'un corpus hétérogène (descriptions d'événements, dates, lieux, tags), permettre des requêtes en langage naturel, et expliquer les sources. Un système RAG permet d'effectuer une recherche sémantique sur les événements (retrieval), puis d'utiliser un modèle de génération (LLM) pour formuler une réponse contextualisée et lisible, tout en limitant la charge sur le modèle (on envoie seulement le contexte pertinent).

- Objectif du POC : Ce que vous cherchez à démontrer (faisabilité technique, valeur métier, performance).

  - Faisabilité technique : Intégration d'OpenAgenda → pipeline de prétraitement → génération d'embeddings via Mistral → index FAISS → orchestration LangChain → API FastAPI.
  - Valeur métier : démontrer que les recommandations retournées sont utiles et exploitables par un utilisateur final (ex. : recherche par genre, date, proximité).
  - Performance : valider la latence et la pertinence (temps moyen de réponse et qualité des réponses pour un jeu de test annoté).

- Périmètre : Zone géographique ciblée, période d’événements, données utilisées.

  - Zone : Rennes Métropole (données OpenAgenda → agenda de Rennes Metropole).
  - Période : configurable via scripts (ex. paramètres `--start-date` / `--end-date` utilisés dans `src.presentation.cli.run_pipeline`).
  - Données : données brutes OpenAgenda (JSON), jeux traités CSV en `Data/processed`, vecteurs d'embeddings en `Data/embeddings`, index FAISS en `Data/index`.

---

2. Architecture du système

- Schéma global (schéma UML) :

  Le projet contient des diagrammes illustrant l'architecture (voir [Documents/COMPONENT_DIAGRAM.md](Documents/COMPONENT_DIAGRAM.md) et [Documents/SEQUENCE_DIAGRAM.md](Documents/SEQUENCE_DIAGRAM.md)).

  Composants principaux :
  - Données entrantes : API OpenAgenda (client : `src.infrastructure.openagenda_client`).
  - Prétraitement : scripts CLI `src.presentation.cli.process_events` qui nettoient et exportent les événements en CSV (`Data/processed`).
  - Chunking & Embeddings : `src.rag.chunker` et `src.rag.embedder` : découpage et appel à l'API d'embeddings (Mistral).
  - Base vectorielle : FAISS via `src.rag.langchain_faiss_indexer` / index files dans `Data/index`.
  - RAG orchestration : `src.domain.services.rag_service` et `src.rag.retriever` (recherche, assemblage du contexte, génération).
  - Exposition : API FastAPI (`src.presentation.api.main`, `src.presentation.api.routes`).

- Technologies utilisées :
  - Langage : Python 3.13+
  - API : FastAPI + Uvicorn
  - Orchestration RAG : LangChain (usage via implémentation locale)
  - Embeddings & LLMs : Mistral AI (via API)
  - Indexation : FAISS
  - Tests : pytest
  - Stockage fichiers : CSV/JSON + index FAISS (fichiers dans `Data/`)

---

3. Préparation et vectorisation des données

- Source de données : API OpenAgenda
  - Client : `src.infrastructure.openagenda_client.py`
  - Paramètres : filtres de dates, mots-clés et paramètres de pagination (définis dans les scripts `src.presentation.cli.process_events` et dans la configuration centrale `config/config.py`).

- Nettoyage :
  - Méthodes : validation et normalisation des champs, suppression des doublons, gestion des dates et des champs texte vides.
  - Exemples d'anomalies corrigées : descriptions manquantes remplacées par un label court, formats de date hétérogènes normalisés en ISO, élimination d'enregistrements non pertinents ou sans localisation.

- Chunking :
  - Raison : limiter la longueur des passages envoyés pour embedding et pour la génération (contrainte de contexte LLM), améliorer granularité de la recherche.
  - Taille choisie : découpage en segments compatibles avec la politique de tokens du LLM et l'usage métier (ex. ~200-400 tokens / chunk). Le paramétrage se trouve dans `src.rag.chunker`.

- Embedding :
  - Modèle utilisé : Mistral embedding API (configuration via variable d'environnement `MISTRAL_API_KEY`).
  - Dimensionnalité : embeddings 1 536D (convention utilisée dans le projet — voir diagramme de séquence). 
  - Logique de batch : génération par lots pour réduire overhead réseau (implémentation dans `src.rag.embedder`).
  - Format des vecteurs : JSON sérialisé en `Data/embeddings/embeddings.json` et importé dans FAISS via `src.rag.langchain_faiss_indexer`.

---

4. Choix du modèle NLP

- Modèle sélectionné :
  - Embeddings : Mistral embedding API.
  - Génération (LLM) : modèle Mistral (via API) ou équivalent compatible LangChain (le code est écrit pour pouvoir remplacer le provider si nécessaire).

- Pourquoi ce modèle ?
  - Critères : compatibilité avec l'API, qualité d'embeddings, disponibilité commerciale (clé API), intégration simple dans le workflow (latence / coût raisonnables pour un POC), compatibilité avec LangChain.

- Prompting (si utilisé) :
  - Strategy : le prompt system fournit les instructions générales (format de réponse), puis le user message contient la question et le contexte extrait (top-k documents). Exemple de structure :

    System: "Tu es un assistant concis et orienté recommandations d'événements. Réponds en français. Cite les sources (événements) utilisées en fin de réponse."

    User/Context: [question utilisateur] + [contexte: extraits texte + métadonnées (titre, date, lieu)]

- Limites du modèle :
  - Hallucinations possibles ; nécessité de garder la génération guidée par le contexte (prompt engineering).
  - Dépendance externe à l'API Mistral (latence, coût, quota).
  - Taille du contexte limitée — nécessité de fournir un contexte restreint et pertinent.

---

5. Construction de la base vectorielle

- FAISS utilisé :
  - Index persisté sur disque (fichiers présents dans `Data/index` : `faiss_index.index`, `faiss_metadata.json`, `index.faiss` selon les runs et formats).
  - Type d'index : configuration paramétrable dans `src.rag.langchain_faiss_indexer` (ex : Flat / IVF selon tests / besoin). Le code supporte la reconstruction via `src.presentation.cli.build_index`.

- Stratégie de persistance :
  - Format de sauvegarde : index FAISS binaire + metadata JSON (pour mapping id → métadonnées). Emplacements : `Data/index/`.
  - Nommage : noms stables (`faiss_index.index`, `faiss_metadata.json`) pour usage automatique par les scripts.

- Métadonnées associées :
  - Pour chaque document / chunk on conserve :
    - `id` interne
    - `title` ou `event_title`
    - `date` (ISO)
    - `location` / `venue`
    - `source_url` (OpenAgenda id / url)
    - `text` (extrait / chunk)
    - autres tags (genre, mots-clés)

---

6. API et endpoints exposés

- Framework utilisé : FastAPI (`src.presentation.api.main`, `src.presentation.api.routes`).

- Endpoints clés :
  - `POST /ask` : requête utilisateur en langage naturel → réponse RAG.
  - `POST /search_raw` : recherche sémantique brute (retourne documents similaires).
  - `POST /update_datas` : déclenche la mise à jour des données (re-fetch depuis OpenAgenda sur plage de dates).
  - `POST /rebuild` : reconstruction de l'index FAISS (batch complet).
  - `GET /evaluate_rag` : déclenche l'évaluation RAG (utilise `src.evaluation.evaluate_rag`).

- Format des requêtes/réponses :
  - `POST /ask` (exemple)

```json
{
  "query": "Quels sont les concerts jazz ce weekend ?"
}
```

Réponse (exemple) :

```json
{
  "answer": "Voici 3 concerts jazz ce weekend à Rennes...",
  "sources": [
    {"id": "123", "title": "Concert X", "date": "2025-06-12", "url": "..."}
  ]
}
```

- Exemple d’appel API (curl) :

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"concerts rock ce weekend"}'
```

- Exemple en Python :

```python
import requests
resp = requests.post('http://localhost:8000/ask', json={"query":"concerts jazz"})
print(resp.json())
```

- Tests effectués et documentés :
  - Tests unitaires et d'intégration via `pytest` (fichiers dans `tests/`).
  - Scripts d'évaluation : `src.evaluation.evaluate_rag.py` et `src.evaluation.build_testset.py`.

- Gestion des erreurs / limitations :
  - Vérification d'entrées via Pydantic dans les routes.
  - Gestion des exceptions liées aux appels API externes (retry/backoff possible dans `openagenda_client`).
  - Contrôle des tailles de contexte pour éviter les erreurs LLM.

---

7. Évaluation du système

- Jeu de test annoté :
  - Emplacement : `src/evaluation/test_questions.json` et `src/evaluation/testset.json`.
  - Nombre d'exemples : voir fichiers (POC — jeu de taille modérée pour validation manuelle).
  - Méthode d'annotation : réponses humaines évaluant pertinence et exactitude ; chaque question est associée à la liste attendue d'événements/sources.

- Métriques d'évaluation :
  - Similarité sémantique (recall@k / precision@k sur documents récupérés).
  - Taux de couverture des réponses (est-ce que la réponse couvre les éléments attendus).
  - Score de satisfaction subjectif (évaluation humaine sur échelle 1–5).

- Résultats obtenus :
 - Résultats obtenus :
  - Analyse quantitative : résultats de l'évaluation automatique (sortie fournie) :
  {'context_precision': 0.4286, 'context_recall': 0.3143, 'answer_relevancy': 0.5466, 'faithfulness': 0.7596}
    - `context_precision`: 0.4286
    - `context_recall`: 0.3143
    - `answer_relevancy`: 0.5466
    - `faithfulness`: 0.7596
  - Interprétation rapide :
    - La précision/recall du module de retrieval est modérée (≈44% / 48.6%) — amélioration possible du chunking, du filtrage ou du type d'index.
    - La pertinence des réponses (`answer_relevancy` ≈52.9%) est moyenne ; le système génère des réponses utiles mais perfectibles.
    - La `faithfulness` (~63.9%) indique que la majorité des réponses restent ancrées dans le contexte, mais des hallucinations subsistent.
  - Analyse qualitative : exemplariser bonnes réponses (contexte bien sélectionné) et cas d'échec (hallucinations, contextes manquants ou mauvaise granularité de chunking).  
    - Recommandations immédiates : augmenter `top_k` pour retrieval lors des tests, ajuster la taille des chunks, tester index FAISS alternatifs (HNSW/IVF), et renforcer le prompt pour réduire les hallucinations.

---

8. Recommandations et perspectives

- Ce qui fonctionne bien :
  - Pipeline de bout en bout (OpenAgenda → embeddings → FAISS → RAG) fonctionnel.
  - Réponses de qualité acceptable pour un POC lorsque le contexte fourni est pertinent.

- Limites du POC :
  - Volumétrie : indexation sur gros volumes non testée à grande échelle (IVF / HNSW à considérer).
  - Performance : latence dépendante du provider embeddings/LLM (Mistral).
  - Coût : usage API externe pour embeddings/LLM peut devenir coûteux en production.

- Améliorations possibles :
  - Ajout d'un cache local pour embeddings fréquents.
  - Évaluer différents types d'index FAISS (HNSW / IVF) pour la scalabilité.
  - Enrichir les métadonnées (tags, popularité) pour filtrage avancé.
  - Pipeline de monitoring (logs, métriques de latence, alerting).
  - Mise en place d'un workflow CI/CD : tests automatisés + build Docker + déploiement.

- Passage en production :
  - Conteneurisation complète (Dockerfile + `docker-compose.yml` fournis) — préparer gestion des secrets, scalabilité, et monitoring.

---

9. Organisation du dépôt GitHub

- Arborescence (fichiers clés et rôle) :

```
PULS_EVENTS/
├── README.md                          # Guide d'utilisation rapide
├── docker-compose.yml                 # Compose pour build/run
├── Dockerfile                         # Image Docker de l'application
├── config/                            # Configs et logger
│   └── config.py
├── src/
│   ├── presentation/
│   │   ├── api/                       # FastAPI
│   │   │   ├── main.py
│   │   │   └── routes.py
│   │   └── cli/                       # Scripts pipeline
│   ├── domain/                        # Logique métier (services)
│   ├── infrastructure/                # Clients externes (OpenAgenda)
│   ├── rag/                           # chunker, embedder, indexer, retriever
│   └── evaluation/                    # scripts d'évaluation
├── Data/                              # Raw / processed / embeddings / index
├── Documents/                         # Diagrammes et documentation technique
└── tests/                             # Test suite (pytest)
```

- Explication rapide :
  - `src.presentation.api` : point d'entrée HTTP
  - `src.presentation.cli` : scripts pour pipeline (récupération, embeddings, build index)
  - `src.domain.services` : orchestration métier (EventService, EmbeddingService, RagService)
  - `src.rag` : composants techniques pour RAG
  - `src.infrastructure` : client OpenAgenda
  - `src.evaluation` : jeux de tests et scripts d'évaluation

---

10. Annexes (exemples)

- Extraits du jeu de test annoté : voir `src/evaluation/test_questions.json`.

- Prompt utilisé (exemple) :

```
System: "Tu es un assistant concis et orienté recommandations d'événements. Réponds en français. Cite les sources en fin de réponse."
User: "{question}"
Context: "{top_k_chunks}"  
```

- Extrait de réponse JSON (exemple) :

```json
{
  "answer": "Voici 2 concerts jazz ce weekend à Rennes...",
  "sources": [
    {"id": "evt_123", "title": "Jazz Night", "date": "2025-06-14", "url": "https://..."}
  ]
}
```

---

Fichiers utiles pour reproduire / exécuter :

```bash
# Build et lancer le container (si Docker installé)
docker-compose up --build

# Lancer localement l'API (dev)
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn src.presentation.api.main:app --reload

# Exécuter pipeline complet
python -m src.presentation.cli.run_pipeline

# Lancer tests
pytest
```

---

Si vous voulez, je peux :
- adapter ce rapport en Français formel pour impression (PDF),
- remplir les résultats d'évaluation quantitatives si vous fournissez les sorties de `src.evaluation.evaluate_rag`,
- ajouter une section « Plan de mise en production » détaillée.
