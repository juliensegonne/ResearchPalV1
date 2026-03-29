import logging
import os
from typing import Callable

from utils import load_config
from retrieval import top_k_similar_indices, mmr_from_documents, score_threshold_filter, rerank, reciprocal_rank_fusion
from generation import get_llm_functions
from query_optimization import self_query, multi_query

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
    "multi_query": False,
}

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

_cfg = load_config(_DEFAULTS, _CONFIG_PATH)

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
RERANK_ENABLED: bool = bool(_cfg["rerank"])
SELF_QUERY_ENABLED: bool = bool(_cfg["self_query"])
MULTI_QUERY_ENABLED: bool = bool(_cfg["multi_query"])
LLM_FN, COMPLETE_FN = get_llm_functions(_cfg["llm"])

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
embedding_model = None
vectorstore = None
doc_embeddings = None
documents = None
doc_metadatas = None


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
    """Reload embeddings/documents/metadatas from vectorstore."""
    global doc_embeddings, documents, doc_metadatas
    if vectorstore is not None:
        data = vectorstore.get(include=["embeddings", "documents", "metadatas"])
        doc_embeddings = data["embeddings"]
        documents = data["documents"]
        doc_metadatas = data["metadatas"]


def close_vectorstore():
    """Ferme le vectorstore pour libérer le verrou SQLite."""
    global vectorstore
    if vectorstore is not None:
        try:
            del vectorstore
        except Exception:
            pass
        vectorstore = None


def clear_vectorstore():
    """Vide tous les documents du vectorstore via l'API ChromaDB."""
    global doc_embeddings, documents, doc_metadatas
    if vectorstore is not None:
        try:
            all_ids = vectorstore.get()["ids"]
            if all_ids:
                vectorstore.delete(ids=all_ids)
            logger.info("Vectorstore vidé via l'API ChromaDB.")
        except Exception as e:
            logger.exception("Échec du vidage ChromaDB: %s", e)
    doc_embeddings = None
    documents = None
    doc_metadatas = None


def is_ready() -> bool:
    """Return True if the pipeline has documents loaded."""
    return doc_embeddings is not None and len(doc_embeddings) > 0


def get_indexed_sources() -> set[str]:
    """Retourne l'ensemble des sources déjà indexées dans ChromaDB."""
    if vectorstore is None:
        return set()
    try:
        data = vectorstore.get(include=["metadatas"])
        return {
            (meta or {}).get("source", "")
            for meta in (data.get("metadatas") or [])
        }
    except Exception:
        return set()


def doc_count() -> int:
    """Return the number of indexed documents."""
    return len(documents) if documents else 0


def _matches_filter(metadata: dict, filter_obj: dict) -> bool:
    """Basic matcher for a subset of ChromaDB-style filters ($eq, $in, $and, $or)."""
    if metadata is None or not isinstance(filter_obj, dict):
        return False

    def _check_op(op: str, actual, expected) -> bool:
        if op == "$eq":
            return actual == expected
        if op == "$ne":
            return actual != expected
        if op == "$in":
            return actual in (expected if isinstance(expected, list) else [expected])
        if op == "$nin":
            return actual not in (expected if isinstance(expected, list) else [expected])
        return False

    def match(obj, meta):
        if not isinstance(obj, dict):
            return False
        for key, value in obj.items():
            # Logical operators
            if key == "$and" and isinstance(value, list):
                if not all(match(v, meta) for v in value):
                    return False
            elif key == "$or" and isinstance(value, list):
                if not any(match(v, meta) for v in value):
                    return False
            # Field-level entry: {"doc_type": "pdf"} or {"doc_type": {"$eq": "pdf"}}
            elif not key.startswith("$"):
                actual = meta.get(key)
                if isinstance(value, dict):
                    # value = {"$eq": "pdf"} or {"$in": ["pdf", "texte"]}
                    for op, expected in value.items():
                        if not _check_op(op, actual, expected):
                            return False
                else:
                    # direct equality: {"doc_type": "pdf"}
                    if actual != value:
                        return False
            else:
                return False
        return True

    try:
        return match(filter_obj, metadata)
    except Exception:
        return False


def _apply_metadata_filter_to_indices(metadatas_list, filter_obj):
    """
    Return list of indices whose metadata matches filter_obj.
    metadatas_list is the list of metadata dicts from ChromaDB.
    """
    matched_indices = []
    for i, meta in enumerate(metadatas_list or []):
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
            matched_idx = _apply_metadata_filter_to_indices(doc_metadatas, metadata_filter)
            if not matched_idx:
                logger.info("Aucun document ne correspond au filtre de métadonnées self-query.")
                return []
            # create filtered lists preserving order
            doc_candidates = [documents[i] for i in matched_idx]
            embeddings_candidates = [doc_embeddings[i] for i in matched_idx]
        except Exception as e:
            logger.warning(f"Erreur lors de l'application du filtre metadata: {e}")
            # fallback to full set

    # Build the list of queries: original + expanded variants
    queries = [semantic_query]
    if MULTI_QUERY_ENABLED:
        try:
            variants = multi_query(semantic_query, COMPLETE_FN)
            queries.extend(variants)
            logger.info(f"multi_query: {len(variants)} variantes ajoutées")
        except Exception as e:
            logger.warning(f"multi_query failed, using original query only: {e}")

    # Retrieve per query, then fuse with RRF if multiple queries
    all_ranked_lists: list[list[str]] = []
    for q in queries:
        q_emb = query_embedding if q == semantic_query else embedding_model.embed_query(q)
        if RETRIEVAL_STRATEGY == "mmr":
            result = mmr_from_documents(
                documents=doc_candidates,
                doc_embeddings=embeddings_candidates,
                query_embedding=q_emb,
                k=RETRIEVAL_K,
                lambda_mult=RETRIEVAL_LAMBDA_MULT,
            )
        elif RETRIEVAL_STRATEGY == "threshold":
            filtered = score_threshold_filter(
                query_embedding=q_emb,
                doc_embeddings=embeddings_candidates,
                documents=doc_candidates,
                threshold=SCORE_THRESHOLD,
            )
            result = [item["text"] for item in filtered]
        else:
            indices = top_k_similar_indices(q_emb, embeddings_candidates, k=RETRIEVAL_K)
            result = [doc_candidates[i] for i in indices]
        all_ranked_lists.append(result)

    # Fuse results
    if len(all_ranked_lists) == 1:
        candidates = all_ranked_lists[0]
    else:
        fused = reciprocal_rank_fusion(all_ranked_lists)
        candidates = fused[:RETRIEVAL_K]
        logger.info(f"RRF fusion: {sum(len(r) for r in all_ranked_lists)} résultats → {len(candidates)} après fusion")

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
