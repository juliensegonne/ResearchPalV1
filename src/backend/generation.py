import logging
import os
from typing import Callable

from google import genai

logger = logging.getLogger("uvicorn.error")

# PROMPT À REVOIR
SYSTEM_PROMPT = (
    "Tu es ResearchPal, un assistant de recherche factuel. "
    "Réponds en te basant UNIQUEMENT sur le contexte fourni et l'historique de conversation valide. "
    "Cite tes sources avec [Source N]. "
    "Si le contexte et l'historique ne permettent pas de répondre, dis-le clairement."
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

    user_message = f"<Contexte>\n{context}\n</Contexte>\n\n<Question>\n{query}\n</Question>"

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

def get_llm_functions(llm_name: str):
    """
    Retourne (llm_fn, complete_fn) selon la config courante.
    LLM totalement découplé du pipeline.
    """
    _LLM_REGISTRY: dict[str, Callable[[str, str, list[dict]], str]] = {
        "gemini": gemini_llm,
    }

    _COMPLETE_REGISTRY: dict[str, Callable[[str], str]] = {
        "gemini": gemini_complete,
    }

    llm_fn = _LLM_REGISTRY.get(llm_name, gemini_llm)
    complete_fn = _COMPLETE_REGISTRY.get(llm_name, gemini_complete)
    return llm_fn, complete_fn
