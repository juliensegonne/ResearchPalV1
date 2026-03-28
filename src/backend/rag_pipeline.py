import json
import logging
import os
from typing import Callable

from retrieval import top_k_similar_indices, mmr_from_documents, score_threshold_filter, rerank
from generation import gemini_llm, gemini_complete
from query_optimization import self_query

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
    "self_query": False,
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

# Résoudre CHROMA_PATH en chemin absolu (module-relative si nécessaire)
_base_dir = os.path.dirname(__file__)
_chroma_cfg_path = _cfg.get("chroma_path", _DEFAULTS["chroma_path"])
if os.path.isabs(_chroma_cfg_path):
    CHROMA_PATH: str = _chroma_cfg_path
else:
    CHROMA_PATH: str = os.path.abspath(os.path.join(_base_dir, _chroma_cfg_path))

RETRIEVAL_STRATEGY: str = _cfg["retrieval_strategy"]
RETRIEVAL_K: int = int(_cfg["retrieval_k"])
RETRIEVAL_LAMBDA_MULT: float = float(_cfg["retrieval_lambda_mult"])
SCORE_THRESHOLD: float = float(_cfg["score_threshold"])
EMBEDDING_MODEL: str = _cfg["embedding_model"]
LLM_FN: Callable[[str, str, list[dict]], str] = _LLM_REGISTRY.get(_cfg["llm"], gemini_llm)
COMPLETE_FN: Callable[[str], str] = _COMPLETE_REGISTRY.get(_cfg["llm"], gemini_complete)
RERANK_ENABLED: bool = bool(_cfg["rerank"])
SELF_QUERY_ENABLED: bool = bool(_cfg["self_query"])

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
        logger.info("Chargement du modèle d'embeddings (%s)...", EMBEDDING_MODEL)
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info("CHROMA_PATH resolved to: %s", CHROMA_PATH)
    if vectorstore is None and os.path.exists(CHROMA_PATH):
        try:
            from langchain_chroma import Chroma
            logger.info("Ouverture de la base ChromaDB à %s", CHROMA_PATH)
            vectorstore = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=embedding_model,
            )
            refresh_docs()
            logger.info("ChromaDB chargée, %d documents trouvés", doc_count() if documents else 0)
        except Exception as e:
            logger.exception("Échec lors de l'ouverture de ChromaDB: %s", e)
            vectorstore = None
    else:
        if vectorstore is None:
            logger.info("Aucune ChromaDB trouvée à %s (existe=%s)", CHROMA_PATH, os.path.exists(CHROMA_PATH))


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


def _matches_filter(metadata: dict, filter_obj: dict) -> bool:
    """Basic matcher for a subset of ChromaDB-style filters ($eq, $in, $and, $or)."""
    if metadata is None or not isinstance(filter_obj, dict):
        return False

    def match(obj, meta):
        # operators
        if not isinstance(obj, dict):
            return False
        for key, value in obj.items():
            if key == "$and" and isinstance(value, list):
                return all(match(v, meta) for v in value)
            if key == "$or" and isinstance(value, list):
                return any(match(v, meta) for v in value)
            if key in {"$eq", "$ne", "$in", "$nin"}:
                # value should be dict with single field: {field: val}
                if not isinstance(value, dict):
                    return False
                for field, expected in value.items():
                    actual = meta.get(field)
                    if key == "$eq" and not (actual == expected):
                        return False
                    if key == "$ne" and not (actual != expected):
                        return False
                    if key == "$in":
                        if isinstance(expected, list):
                            if actual not in expected:
                                return False
                        else:
                            if actual != expected:
                                return False
                    if key == "$nin":
                        if isinstance(expected, list):
                            if actual in expected:
                                return False
                        else:
                            if actual == expected:
                                return False
                return True
            # direct field match like {"doc_type": {"$eq": "pdf"}} or {"doc_type": "pdf"}
            if key in metadata:
                # value can be operator dict or direct expected value
                if isinstance(value, dict):
                    # recurse to handle {"doc_type": {"$eq": "pdf"}}
                    return match({list(value.keys())[0]: {key: list(value.values())[0]}}, metadata)
                else:
                    return metadata.get(key) == value
        # fallback false
        return False

    try:
        return match(filter_obj, metadata)
    except Exception:
        return False


def _apply_metadata_filter_to_indices(documents_list, filter_obj):
    """
    Return list of indices that match filter_obj.
    documents_list can be list of dicts (with metadata) or strings (no metadata).
    We try common metadata placements: doc.get("metadata"), doc itself if dict,
    or doc.get("source"/"doc_type"/"ingestion_date").
    """
    matched_indices = []
    for i, doc in enumerate(documents_list):
        meta = None
        if isinstance(doc, dict):
            meta = doc.get("metadata") or {k: v for k, v in doc.items() if k in {"source", "doc_type", "ingestion_date"}}
            # if metadata is empty, maybe doc dict is already a metadata-like object
            if not meta:
                meta = {k: v for k, v in doc.items() if k not in {"text", "page", "content"}}
        # strings cannot be filtered
        if _matches_filter(meta, filter_obj):
            matched_indices.append(i)
    return matched_indices


def retrieve(query: str) -> list[str]:
    """Embed the query, retrieve relevant documents, then rerank."""
    init_models()

    # Self-query: obtain semantic_query and optional metadata_filter
    semantic_query = query
    metadata_filter = None
    if SELF_QUERY_ENABLED:
        try:
            sq = self_query(query, COMPLETE_FN)
            semantic_query = sq.get("semantic_query") or query
            metadata_filter = sq.get("metadata_filter")
            logger.info(f"self-query -> semantic='{semantic_query}', filter={metadata_filter}")
        except Exception as e:
            logger.warning(f"self-query failed, using original query: {e}")

    query_embedding = embedding_model.embed_query(semantic_query)

    # If a metadata_filter is present, try to restrict documents/embeddings
    doc_candidates = documents
    embeddings_candidates = doc_embeddings
    if metadata_filter and documents:
        try:
            matched_idx = _apply_metadata_filter_to_indices(documents, metadata_filter)
            if not matched_idx:
                logger.info("Aucun document ne correspond au filtre de métadonnées self-query.")
                return []
            # create filtered lists preserving order
            doc_candidates = [documents[i] for i in matched_idx]
            embeddings_candidates = [doc_embeddings[i] for i in matched_idx]
        except Exception as e:
            logger.warning(f"Erreur lors de l'application du filtre metadata: {e}")
            # fallback to full set

    if RETRIEVAL_STRATEGY == "mmr":
        candidates = mmr_from_documents(
            documents=doc_candidates,
            doc_embeddings=embeddings_candidates,
            query_embedding=query_embedding,
            k=RETRIEVAL_K,
            lambda_mult=RETRIEVAL_LAMBDA_MULT,
        )
    elif RETRIEVAL_STRATEGY == "threshold":
        filtered = score_threshold_filter(
            query_embedding=query_embedding,
            doc_embeddings=embeddings_candidates,
            documents=doc_candidates,
            threshold=SCORE_THRESHOLD,
        )
        candidates = [item["text"] for item in filtered]
    else:
        indices = top_k_similar_indices(query_embedding, embeddings_candidates, k=RETRIEVAL_K)
        candidates = [doc_candidates[i] for i in indices]

    # Reranking cross-encoder
    if RERANK_ENABLED and candidates:
        ranked = rerank(query=semantic_query, documents=candidates, k=RETRIEVAL_K)
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
