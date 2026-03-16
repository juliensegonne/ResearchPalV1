from langchain_community.utils.math import cosine_similarity as _lc_cosine
import numpy as np
from typing import Iterable, List, Sequence


def cosine_similarity(a, b):
    """
    Robust wrapper autour de la fonction de langchain si disponible.
    Gère les entrées 1D ([..]) et 2D ([[..]]) en convertissant quand nécessaire.
    Retourne un float (similarité entre deux vecteurs).
    """
    try:
        # normaliser les cas [[v]] -> [v] pour la fonction langchain si présente
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        if _lc_cosine is not None:
            a_use = a_arr[0].tolist() if a_arr.ndim == 2 and a_arr.shape[0] == 1 else a_arr.tolist()
            b_use = b_arr[0].tolist() if b_arr.ndim == 2 and b_arr.shape[0] == 1 else b_arr.tolist()
            return float(_lc_cosine(a_use, b_use))
    except Exception:
        # fallback vers numpy si langchain échoue
        pass

    # fallback numpy : assure qu'on compare deux vecteurs 1D
    a_vec = np.asarray(a, dtype=float)
    b_vec = np.asarray(b, dtype=float)
    if a_vec.ndim == 2 and a_vec.shape[0] == 1:
        a_vec = a_vec[0]
    if b_vec.ndim == 2 and b_vec.shape[0] == 1:
        b_vec = b_vec[0]

    if a_vec.shape != b_vec.shape:
        raise ValueError("Les embeddings doivent avoir la même taille")

    norma = np.linalg.norm(a_vec)
    normb = np.linalg.norm(b_vec)
    if norma == 0 or normb == 0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / (norma * normb))


def mmr_from_embeddings(
    doc_embeddings: Sequence[Iterable[float]],
    query_embedding: Iterable[float],
    k: int = 5,
    lambda_mult: float = 0.5
) -> List[int]:
    """
    Maximum Marginal Relevance (MMR) sur une liste d'embeddings.
    Retourne la liste des indices des documents sélectionnés (ordre de sélection).
    - doc_embeddings : sequence de vecteurs (n_docs, dim)
    - query_embedding : vecteur (dim,)
    - k : nombre de documents à sélectionner
    - lambda_mult : équilibre entre pertinence (query) et diversité (docs), dans [0,1]
    """
    if k <= 0:
        return []
    
    docs = np.asarray([np.asarray(v, dtype=float) for v in doc_embeddings])
    query = np.asarray(list(query_embedding), dtype=float)

    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]

    if docs.ndim != 2:
        raise ValueError("doc_embeddings doit avoir une forme (n_docs, dim)")
    if query.ndim != 1 or docs.shape[1] != query.shape[0]:
        raise ValueError("Dimensions incompatibles entre docs et query")

    n = docs.shape[0]
    k = min(k, n)

    # similarité doc-query
    doc_norms = np.linalg.norm(docs, axis=1)
    q_norm = np.linalg.norm(query)
    # éviter divisions par zéro
    doc_norms_safe = np.where(doc_norms == 0, 1.0, doc_norms)
    q_norm_safe = q_norm if q_norm != 0 else 1.0
    sim_to_query = (docs @ query) / (doc_norms_safe * q_norm_safe)

    # matrice similarité entre docs
    denom = np.outer(doc_norms_safe, doc_norms_safe)
    sim_docs = (docs @ docs.T) / denom
    # clip pour éviter légères dérives numériquement >1 ou <-1
    sim_docs = np.clip(sim_docs, -1.0, 1.0)

    selected: List[int] = []
    candidates = set(range(n))

    # initial : le doc le plus similaire à la query
    first = int(np.argmax(sim_to_query))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < k:
        mmr_scores = {}
        for c in candidates:
            relevance = sim_to_query[c]
            # diversité = max similitude du candidat avec les docs déjà sélectionnés
            max_sim_selected = max(sim_docs[c, s] for s in selected) if selected else 0.0
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_selected
            mmr_scores[c] = mmr_score
        # choisir le candidat avec le score MMR maximal
        next_idx = max(mmr_scores, key=mmr_scores.get)
        selected.append(next_idx)
        candidates.remove(next_idx)

    return selected


def mmr_from_documents(
    documents: Sequence[str],
    doc_embeddings: Sequence[Iterable[float]],
    query_embedding: Iterable[float],
    k: int = 5,
    lambda_mult: float = 0.5
) -> List[str]:
    """
    Applique MMR et retourne les documents (texte) sélectionnés.
    """
    indices = mmr_from_embeddings(doc_embeddings, query_embedding, k=k, lambda_mult=lambda_mult)
    return [documents[i] for i in indices]


def top_k_similar_indices(
    query_embedding,
    doc_embeddings,
    k: int = 5
) -> List[int]:
    """
    Retourne les indices des top-k documents selon la similarité cosinus
    entre `query_embedding` et `doc_embeddings`.
    Utilise numpy (vecteur) pour être rapide et gère les shapes (1D / 2D).
    """
    docs = np.asarray([np.asarray(v, dtype=float) for v in doc_embeddings])
    query = np.asarray(list(query_embedding), dtype=float)

    # gérer query donnée sous forme [[...]]
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]

    if docs.ndim != 2:
        raise ValueError("doc_embeddings doit avoir la forme (n_docs, dim)")
    if query.ndim != 1 or docs.shape[1] != query.shape[0]:
        raise ValueError("Dimensions incompatibles entre docs et query")

    # calcul vectorisé similarité cosinus
    doc_norms = np.linalg.norm(docs, axis=1)
    q_norm = np.linalg.norm(query)
    doc_norms_safe = np.where(doc_norms == 0, 1.0, doc_norms)
    q_norm_safe = q_norm if q_norm != 0 else 1.0

    sims = (docs @ query) / (doc_norms_safe * q_norm_safe)
    sims = np.clip(sims, -1.0, 1.0)

    k = min(int(k), docs.shape[0])
    topk_idx = np.argsort(-sims)[:k].tolist()
    return topk_idx

