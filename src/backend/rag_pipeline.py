import json
import logging
import os
from typing import Callable

from retrieval import top_k_similar_indices, mmr_from_documents, score_threshold_filter, rerank
from generation import gemini_llm, gemini_complete

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "chroma_path": "./chroma_db",
    "retrieval_strategy": "cosine",
    "retrieval_k": 5,
    "retrieval_lambda_mult": 0.5,
    "score_threshold": 0.5,
    "embedding_model": "all-mpnet-base-v2",
    "llm": "gemini",
    "rerank": False,
}

_LLM_REGISTRY: dict[str, Callable[[str, str, list[dict]], str]] = {
    "gemini": gemini_llm,
}

_COMPLETE_REGISTRY: dict[str, Callable[[str], str]] = {
    "gemini": gemini_complete,
}

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def _load_config() -> dict:
    """Load config from config.json, falling back to defaults."""
    cfg = dict(_DEFAULTS)
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH) as f:
                user_cfg = json.load(f)
            cfg.update(user_cfg)
            logger.info(f"⚙️ Configuration chargée depuis {_CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lecture config, valeurs par défaut utilisées : {e}")
    else:
        logger.info("⚙️ Aucun config.json trouvé, valeurs par défaut utilisées")
    return cfg


_cfg = _load_config()

CHROMA_PATH: str = _cfg["chroma_path"]
RETRIEVAL_STRATEGY: str = _cfg["retrieval_strategy"]
RETRIEVAL_K: int = int(_cfg["retrieval_k"])
RETRIEVAL_LAMBDA_MULT: float = float(_cfg["retrieval_lambda_mult"])
SCORE_THRESHOLD: float = float(_cfg["score_threshold"])
EMBEDDING_MODEL: str = _cfg["embedding_model"]
LLM_FN: Callable[[str, str, list[dict]], str] = _LLM_REGISTRY.get(_cfg["llm"], gemini_llm)
COMPLETE_FN: Callable[[str], str] = _COMPLETE_REGISTRY.get(_cfg["llm"], gemini_complete)
RERANK_ENABLED: bool = bool(_cfg["rerank"])

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
embedding_model = None
vectorstore = None
doc_embeddings = None
documents = None


def init_models():
    """Lazy-load embedding model + vectorstore."""
    global embedding_model, vectorstore
    if embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if vectorstore is None and os.path.exists(CHROMA_PATH):
        from langchain_chroma import Chroma
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model,
        )
        refresh_docs()


def refresh_docs():
    """Reload embeddings/documents from vectorstore."""
    global doc_embeddings, documents
    if vectorstore is not None:
        data = vectorstore.get(include=["embeddings", "documents"])
        doc_embeddings = data["embeddings"]
        documents = data["documents"]


def reset_state():
    """Reset all RAG state (after clearing the database)."""
    global vectorstore, doc_embeddings, documents
    vectorstore = None
    doc_embeddings = None
    documents = None


def is_ready() -> bool:
    """Return True if the pipeline has documents loaded."""
    return doc_embeddings is not None and len(doc_embeddings) > 0


def doc_count() -> int:
    """Return the number of indexed documents."""
    return len(documents) if documents else 0


def retrieve(query: str) -> list[str]:
    """Embed the query, retrieve relevant documents, then rerank."""
    init_models()
    query_embedding = embedding_model.embed_query(query)

    if RETRIEVAL_STRATEGY == "mmr":
        candidates = mmr_from_documents(
            documents=documents,
            doc_embeddings=doc_embeddings,
            query_embedding=query_embedding,
            k=RETRIEVAL_K,
            lambda_mult=RETRIEVAL_LAMBDA_MULT,
        )
    elif RETRIEVAL_STRATEGY == "threshold":
        filtered = score_threshold_filter(
            query_embedding=query_embedding,
            doc_embeddings=doc_embeddings,
            documents=documents,
            threshold=SCORE_THRESHOLD,
        )
        candidates = [item["text"] for item in filtered]
    else:
        indices = top_k_similar_indices(query_embedding, doc_embeddings, k=RETRIEVAL_K)
        candidates = [documents[i] for i in indices]

    # Reranking cross-encoder
    if RERANK_ENABLED and candidates:
        ranked = rerank(query=query, documents=candidates, k=RETRIEVAL_K)
        return [item["text"] for item in ranked]
    return candidates


def generate_answer(
    query: str,
    context: str,
    sources: list[str],
    conversation_history: list[dict],
) -> str:
    """Generate an answer using the configured LLM function.

    Falls back to formatted sources if the LLM call fails.
    """
    try:
        return LLM_FN(query, context, conversation_history)
    except Exception as e:
        logger.error(f"❌ Erreur LLM : {e}")

    # Fallback: formatted retrieval results
    lines = ["Voici les passages les plus pertinents trouvés :\n"]
    for i, src in enumerate(sources, 1):
        lines.append(f"**[Source {i}]** {src}\n")
    return "\n".join(lines)
