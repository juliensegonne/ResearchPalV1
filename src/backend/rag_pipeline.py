import logging
import os

from retrieval import top_k_similar_indices, mmr_from_documents, score_threshold_filter, rerank

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PATH = "./chroma_db"
RETRIEVAL_STRATEGY = "cosine"   # "cosine" | "mmr" | "threshold"
RETRIEVAL_K = 5
RETRIEVAL_LAMBDA_MULT = 0.5
SCORE_THRESHOLD = 0.5

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
        embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

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
    if candidates:
        ranked = rerank(query=query, documents=candidates, k=RETRIEVAL_K)
        return [item["text"] for item in ranked]
    return candidates


def generate_answer(
    query: str,
    context: str,
    sources: list[str],
    conversation_history: list[dict],
) -> str:
    """Call Gemini LLM. Falls back to a formatted context summary."""
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("⚠️ Aucune clé API trouvée (GEMINI_API_KEY)")
        if api_key:
            from google import genai

            client = genai.Client(api_key=api_key)

            system_prompt = (
                "Tu es ResearchPal, un assistant de recherche factuel. "
                "Réponds en te basant UNIQUEMENT sur le contexte fourni. "
                "Cite tes sources avec [Source N]. "
                "Si le contexte ne permet pas de répondre, dis-le clairement."
            )

            history_parts = []
            for msg in conversation_history[-20:]:
                role = "user" if msg["role"] == "user" else "model"
                history_parts.append(genai.types.Content(role=role, parts=[genai.types.Part(text=msg["content"])]))

            user_message = f"Contexte :\n{context}\n\nQuestion : {query}"

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    *history_parts,
                    genai.types.Content(role="user", parts=[genai.types.Part(text=user_message)]),
                ],
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                ),
            )
            return response.text
    except Exception as e:
        logger.error(f"❌ Erreur LLM : {e}")

    # Fallback: formatted retrieval results
    lines = ["Voici les passages les plus pertinents trouvés :\n"]
    for i, src in enumerate(sources, 1):
        lines.append(f"**[Source {i}]** {src}\n")
    return "\n".join(lines)
