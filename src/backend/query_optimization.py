"""
query_optimization.py — Optimisation de requête par self-query.

Utilise Gemini pour décomposer une requête utilisateur en :
  1. Une requête sémantique épurée (pour la recherche vectorielle)
  2. Des filtres de métadonnées (pour ChromaDB where clause)
"""

import json
import logging
from typing import Callable

logger = logging.getLogger("uvicorn.error")

# Métadonnées connues dans le vectorstore
METADATA_SCHEMA = {
    "source": {
        "type": "string",
        "description": "Chemin du fichier ou URL d'origine du document",
    },
    "doc_type": {
        "type": "string",
        "description": "Type de document",
    },
    "ingestion_date": {
        "type": "string",
        "description": "Date d'ingestion au format YYYY-MM-DD",
    },
}

SELF_QUERY_PROMPT = """\
Tu es un analyseur de requêtes pour un moteur de recherche documentaire.

Ton rôle : décomposer la requête utilisateur en deux parties :
1. **semantic_query** : la question reformulée pour une recherche sémantique (sans les critères de filtrage).
2. **metadata_filter** : un objet de filtre ChromaDB (format `where`) basé sur les métadonnées disponibles. `null` si aucun filtre ne s'applique.

Métadonnées disponibles :
{schema}

Règles :
- Ne génère un filtre QUE si la requête mentionne explicitement un critère filtrable (type de document, source, date).
- Utilise les opérateurs ChromaDB : $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or.
- Réponds UNIQUEMENT avec un objet JSON valide, sans markdown, sans explication.

Format de réponse :
{{"semantic_query": "...", "metadata_filter": {{...}} | null}}

Exemples :
- Requête : "Que disent les PDF sur le machine learning ?"
  → {{"semantic_query": "machine learning", "metadata_filter": {{"doc_type": {{"$eq": "pdf"}}}}}}

- Requête : "Résume les documents ingérés après le 2025-01-01"
  → {{"semantic_query": "résumé des documents", "metadata_filter": {{"ingestion_date": {{"$gte": "2025-01-01"}}}}}}

- Requête : "Comment fonctionne l'attention dans les transformers ?"
  → {{"semantic_query": "fonctionnement attention transformers", "metadata_filter": null}}

Requête utilisateur : {query}
"""


def self_query(query: str, complete_fn: Callable[[str], str]) -> dict:
    """
    Décompose une requête utilisateur via un LLM en :
      - semantic_query : str — requête optimisée pour la recherche vectorielle
      - metadata_filter : dict | None — filtre ChromaDB `where`

    Retourne la requête originale sans filtre si l'appel échoue.
    """
    fallback = {"semantic_query": query, "metadata_filter": None}

    try:
        schema_text = json.dumps(METADATA_SCHEMA, indent=2, ensure_ascii=False)
        prompt = SELF_QUERY_PROMPT.format(schema=schema_text, query=query)

        raw = complete_fn(prompt).strip()
        # Nettoyer d'éventuels blocs markdown ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        result = json.loads(raw)

        semantic_query = result.get("semantic_query")
        metadata_filter = result.get("metadata_filter")

        if not semantic_query or not isinstance(semantic_query, str):
            logger.warning("⚠️ Self-query : semantic_query invalide, fallback")
            return fallback

        # Valider que le filtre ne référence que des champs connus
        if metadata_filter is not None:
            _validate_filter_keys(metadata_filter)

        logger.info(f"🔍 Self-query : '{query}' → semantic='{semantic_query}', filter={metadata_filter}")
        return {"semantic_query": semantic_query, "metadata_filter": metadata_filter}

    except json.JSONDecodeError as e:
        logger.error(f"❌ Self-query JSON invalide : {e}")
    except Exception as e:
        logger.error(f"❌ Self-query erreur : {e}")

    return fallback


def _validate_filter_keys(filter_obj: dict) -> None:
    """
    Vérifie récursivement que le filtre ne contient que des clés
    de métadonnées connues ou des opérateurs ChromaDB.
    Lève ValueError si une clé inconnue est détectée.
    """
    operators = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$and", "$or"}
    for key, value in filter_obj.items():
        if key in operators:
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        _validate_filter_keys(item)
            elif isinstance(value, dict):
                _validate_filter_keys(value)
        elif key in METADATA_SCHEMA:
            if isinstance(value, dict):
                _validate_filter_keys(value)
        else:
            raise ValueError(f"Clé de filtre inconnue : '{key}'")
