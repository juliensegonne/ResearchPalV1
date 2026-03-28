# ResearchPal v1 — Pipeline RAG de base

> Assistant de recherche personnel basé sur une architecture RAG (Retrieval-Augmented Generation).

## Architecture

```
ResearchPalV1/
├── docker-compose.yml     # Orchestration Docker
├── data/                  # Corpus de documents (PDF, TXT, MD)
├── src/
│   ├── backend/           # API FastAPI + pipeline RAG (Python)
│   │   ├── Dockerfile
│   │   ├── api.py         # Serveur API REST
│   │   ├── rag_pipeline.py # Pipeline RAG (retrieval + génération LLM)
│   │   ├── config.json    # Configuration du pipeline RAG (optionnel)
│   │   ├── indexation_config.json # Configuration de l'indexation (optionnel)
│   │   ├── generation.py  # Implémentation LLM (Gemini)
│   │   ├── indexation.py  # Ingestion & indexation (ChromaDB)
│   │   ├── retrieval.py   # Stratégies de retrieval (cosinus, MMR, reranking, filtrage)
│   │   ├── query_optimization.py  # Optimisation de requête par self-query (Gemini)
│   │   └── main.py        # Script CLI d'évaluation
│   └── frontend/          # Interface Angular
│       ├── Dockerfile
│       ├── nginx.conf     # Reverse proxy pour Docker
│       └── src/app/
│           ├── chat/       # Composant de chat conversationnel
│           └── documents/  # Gestion des documents
└── README.md
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| LLM | Gemini 2.5 Flash (Google) |
| Embeddings | Sentence Transformers (`all-mpnet-base-v2`) |
| Base vectorielle | ChromaDB |
| Backend | FastAPI + Uvicorn |
| Frontend | Angular 21 |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) |

## Prérequis

- **Python 3.10+** avec `pip`
- **Node.js 20+** avec `npm`

## Installation & lancement

### 1. Backend (API Python)

```bash
cd src/backend

# Créer et activer l'environnement virtuel
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt

# Configurer la clé API Gemini
export GEMINI_API_KEY="votre-clé-gemini"

# Lancer le serveur
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Le backend sera accessible sur **http://localhost:8000**.
Documentation Swagger : http://localhost:8000/docs

> Sans clé API, le chat retourne directement les passages récupérés sans reformulation par un LLM.

### 2. Frontend (Angular)

```bash
# Dans un nouveau terminal
cd src/frontend
npm install
npx ng serve
```

L'interface sera accessible sur **http://localhost:4200**.

## Utilisation

1. Ouvrir **http://localhost:4200** dans un navigateur.
2. Aller dans **📄 Documents** pour téléverser des fichiers (PDF, TXT, MD) ou ajouter une URL.
3. Cliquer sur **Ingérer les documents** pour indexer les fichiers dans la base vectorielle.
4. Revenir sur **💬 Chat** et poser des questions en langage naturel.
5. Les réponses incluent les **sources citées** consultables via le bouton dédié.
6. Utiliser **Vider la base** pour réinitialiser la base vectorielle si nécessaire.

## Points d'API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/health` | État du serveur |
| `GET` | `/api/documents` | Liste des fichiers et URLs indexées |
| `POST` | `/api/documents/upload` | Téléverser un fichier |
| `POST` | `/api/documents/add-url` | Ajouter une URL |
| `POST` | `/api/documents/ingest` | Indexer tous les documents dans ChromaDB |
| `DELETE` | `/api/documents/clear` | Vider la base vectorielle |
| `POST` | `/api/chat` | Requête RAG conversationnelle |
| `POST` | `/api/search` | Recherche sans LLM |
| `GET` | `/api/history` | Historique de conversation |
| `DELETE` | `/api/history` | Effacer l'historique |

## Modules backend

| Fichier | Rôle |
|---------|------|
| `api.py` | Serveur FastAPI, endpoints REST |
| `rag_pipeline.py` | Pipeline RAG : config depuis `config.json`, retrieval (cosinus/MMR/seuil + reranking) et génération (LLM interchangeable) |
| `config.json` | Configuration du pipeline RAG : stratégie, seuils, modèle d'embeddings, LLM (optionnel, valeurs par défaut intégrées) |
| `indexation_config.json` | Configuration de l'indexation : chunk_size, chunk_overlap, séparateurs (optionnel, valeurs par défaut intégrées) |
| `generation.py` | Implémentation Gemini 2.5 Flash (`gemini_llm`) — interchangeable |
| `indexation.py` | Chargement de fichiers (PDF/TXT/MD/URL), chunking, stockage ChromaDB |
| `retrieval.py` | Similarité cosinus, MMR, filtrage par métadonnées, reranking cross-encoder, seuil de score |
| `query_optimization.py` | Self-query : décomposition requête → requête sémantique + filtres métadonnées via Gemini |
| `main.py` | Script CLI pour évaluation cosinus vs MMR |

## Lancement avec Docker

### Prérequis

- **Docker** et **Docker Compose** installés
- Une **clé API Gemini** valide

### Lancement

```bash
# Configurer la clé API
export GEMINI_API_KEY="votre-clé-gemini"

# Construire et lancer les conteneurs
docker compose up --build
```

L'application sera accessible sur **http://localhost:4200**.

### Architecture Docker

| Service | Image | Port | Détails |
|---------|-------|------|---------|
| `backend` | Python 3.12 + FastAPI | 8000 | API REST, pipeline RAG |
| `frontend` | Node 22 (build) → Nginx (serve) | 4200 → 80 | SPA Angular, reverse proxy `/api/` → backend |

- Le frontend Nginx redirige les appels `/api/` vers le backend.
- Les données ChromaDB et les fichiers uploadés sont persistés via des volumes Docker.
- En production (Docker), l'API est accédée via le proxy Nginx (`/api`), ce qui évite les problèmes CORS.

### Arrêt

```bash
docker compose down
# Pour supprimer aussi les volumes (données persistantes) :
docker compose down -v
```