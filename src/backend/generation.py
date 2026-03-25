import logging
import os

logger = logging.getLogger("uvicorn.error")

# PROMPT À REVOIR
SYSTEM_PROMPT = (
    "Tu es ResearchPal, un assistant de recherche factuel. "
    "Réponds en te basant UNIQUEMENT sur le contexte fourni. "
    "Cite tes sources avec [Source N]. "
    "Si le contexte ne permet pas de répondre, dis-le clairement."
    #question refinement
    "Quand une question est posée, suggère une version améliorée de la question avec plus de vocabulaires et de verbes spécifiques, puis réponds à la version améliorée."
)


def gemini_llm(query: str, context: str, conversation_history: list[dict]) -> str:
    """Generate an answer using Gemini 2.5 Flash."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("⚠️ Aucune clé API trouvée (GEMINI_API_KEY)")
        raise RuntimeError("GEMINI_API_KEY non configurée")

    from google import genai

    client = genai.Client(api_key=api_key)

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
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
        ),
    )
    return response.text


def gemini_complete(prompt: str) -> str:
    """Simple text completion using Gemini 2.5 Flash."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY non configurée")

    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=0.0),
    )
    return response.text
