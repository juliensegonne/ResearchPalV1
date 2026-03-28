import os
import sys
import shutil
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure task modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from indexation import load_and_chunk, store_in_chroma
import rag_pipeline as rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)
    rag.init_models()
    yield


app = FastAPI(
    title="ResearchPal API",
    description=(
        "API RAG (Retrieval-Augmented Generation) pour ResearchPal.\n\n"
        "- **Documents** : upload, ingestion et listing de fichiers (PDF, TXT, MD) et URLs.\n"
        "- **Recherche** : similarité cosinus ou MMR, sans LLM.\n"
        "- **Chat** : requête RAG conversationnelle avec Gemini.\n"
        "- **Historique** : consultation et suppression de l'historique de conversation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Santé", "description": "État du serveur"},
        {"name": "Documents", "description": "Gestion des fichiers et ingestion"},
        {"name": "Recherche", "description": "Recherche sémantique (cosinus / MMR)"},
        {"name": "Chat", "description": "Chat RAG avec LLM"},
        {"name": "Historique", "description": "Historique de conversation"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
DATA_DIR = "data"

# Conversation history: list of {role, content, sources}
conversation_history: list[dict] = []

logger = logging.getLogger("researchpal")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str


class ChatMessage(BaseModel):
    role: str          # "user" | "assistant"
    content: str
    sources: list[str] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["Santé"], summary="Vérifier l'état du serveur")
def health():
    has_db = os.path.exists(rag.CHROMA_PATH)
    return {"status": "ok", "has_database": has_db, "document_count": rag.doc_count()}


# ---- Document management --------------------------------------------------

@app.post("/api/documents/upload", tags=["Documents"], summary="Téléverser un fichier")
async def upload_document(file: UploadFile = File(...)):
    """Upload a file (PDF, TXT, MD) into the data directory."""
    allowed = {".pdf", ".txt", ".md"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Type de fichier non supporté : {ext}")

    safe_name = os.path.basename(file.filename or "upload")
    dest = os.path.join(DATA_DIR, safe_name)
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    return {"message": f"Fichier '{safe_name}' téléversé avec succès.", "filename": safe_name}


async def _ingest_worker(data_dir: str, urls: list | None = None):
    """Worker asynchrone : exécute les fonctions bloquantes dans un thread."""
    try:
        logger.info("Démarrage du worker d'ingestion (background)...")
        already_indexed = rag.get_indexed_sources()
        chunks = await asyncio.to_thread(load_and_chunk, data_dir, urls, already_indexed)
        if not chunks:
            logger.info("Aucun chunk produit par load_and_chunk.")
            return
        # store_in_chroma (bloquant) en thread
        await asyncio.to_thread(store_in_chroma, chunks, rag.CHROMA_PATH)
        # réinitialiser / rafraîchir l'état du pipeline
        await asyncio.to_thread(rag.init_models)
        await asyncio.to_thread(rag.refresh_docs)
        logger.info("Ingestion terminée.")
    except Exception as e:
        logger.exception("Erreur dans le worker d'ingestion: %s", e)


@app.post("/api/documents/add-url", tags=["Documents"], summary="Ingérer une page web par URL")
async def add_url(url: str = Form(...)):
    """Enregistre l'URL puis démarre l'ingestion en tâche de fond."""
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL invalide")
    # sauvegarde légère (facultative) : on peut stocker l'URL quelque part si besoin
    # Démarrer l'ingestion en background et retourner immédiatement
    asyncio.create_task(_ingest_worker(DATA_DIR, [url]))
    return {"message": f"Ingestion de l'URL planifiée en arrière-plan : {url}"}


@app.get("/api/documents", tags=["Documents"], summary="Lister les fichiers et URLs disponibles")
def list_documents():
    """List files in the data directory and ingested URLs from the vectorstore."""
    items = []

    # 1. Fichiers locaux
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            items.append({
                "name": f,
                "type": "file",
                "size": os.path.getsize(path),
                "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
            })

    # 2. URLs ingérées (extraites des métadonnées ChromaDB)
    if rag.vectorstore is not None:
        try:
            data = rag.vectorstore.get(include=["metadatas"])
            seen_urls: set[str] = set()
            for meta in (data.get("metadatas") or []):
                source = (meta or {}).get("source", "")
                logger.info(f"Vérification source pour URL : {source}")
                if source.startswith(("http://", "https://")) and source not in seen_urls:
                    seen_urls.add(source)
                    items.append({
                        "name": source,
                        "type": "url",
                        "size": 0,
                        "modified": (meta or {}).get("ingestion_date", ""),
                    })
        except Exception:
            pass

    return items


@app.post("/api/documents/ingest", tags=["Documents"], summary="Indexer tous les fichiers du dossier data/")
async def ingest_documents():
    """Démarre l'ingestion de tout le dossier data/ en tâche de fond (non bloquant)."""
    # quick check existence
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise HTTPException(400, "Aucun document trouvé dans le dossier data/")
    asyncio.create_task(_ingest_worker(DATA_DIR, None))
    return {"message": "Ingestion démarrée en arrière-plan."}


# ---- Search / Chat --------------------------------------------------------

@app.delete("/api/documents/clear", tags=["Documents"], summary="Vider la base de données vectorielle")
def clear_database():
    """Delete the ChromaDB database and all uploaded files."""
    rag.clear_vectorstore()
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            os.remove(path)
    return {"message": "Base de données et fichiers supprimés."}


# ---- Search / Chat --------------------------------------------------------

@app.post("/api/search", tags=["Recherche"], summary="Recherche sémantique (sans LLM)")
def search(req: QueryRequest):
    """Search documents using the configured retrieval strategy (no LLM)."""
    if not rag.is_ready():
        raise HTTPException(400, "La base de données est vide. Veuillez d'abord ingérer des documents.")

    retrieved = rag.retrieve(req.query)

    results = [{"rank": rank, "text": text} for rank, text in enumerate(retrieved, 1)]
    return {"query": req.query, "strategy": rag.RETRIEVAL_STRATEGY, "results": results}


@app.post("/api/chat", tags=["Chat"], summary="Chat RAG conversationnel")
def chat(req: QueryRequest):
    """
    RAG chat endpoint: retrieve relevant chunks, build a prompt and return
    the answer with cited sources.

    If no LLM is configured, returns the retrieved context as-is so the
    frontend still works.
    """
    if not rag.is_ready():
        raise HTTPException(400, "Base vide — ingérez des documents d'abord.")

    retrieved = rag.retrieve(req.query)

    sources = [t[:300] for t in retrieved]

    # Build context for LLM
    context_block = "\n\n".join(
        f"[Source {i+1}] {chunk}" for i, chunk in enumerate(retrieved)
    )

    # Try to call an LLM; fallback to returning raw context
    answer = rag.generate_answer(req.query, context_block, sources, conversation_history)

    # Store in history (keep last 10 turns = 20 messages)
    conversation_history.append({"role": "user", "content": req.query, "sources": []})
    conversation_history.append({"role": "assistant", "content": answer, "sources": sources})
    del conversation_history[:-20]

    return {
        "answer": answer,
        "sources": sources,
        "history": conversation_history,
    }


@app.get("/api/history", tags=["Historique"], summary="Consulter l'historique")
def get_history():
    return conversation_history


@app.delete("/api/history", tags=["Historique"], summary="Effacer l'historique")
def clear_history():
    conversation_history.clear()
    return {"message": "Historique effacé."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
