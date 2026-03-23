import logging
import os
from indexation import load_and_chunk, store_in_chroma
from retrieval import top_k_similar_indices, mmr_from_documents
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger("researchpal.main")

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    CHROMA_PATH = "./chroma_db"
    
    logger.info("--- 1. Initialisation du système ---")
    # On définit le modèle (doit être le même que celui utilisé à l'ingestion)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Vérification de l'existence de la base
    if not os.path.exists(CHROMA_PATH):
        logger.error(f"❌ Erreur : La base de données à '{CHROMA_PATH}' est introuvable.")
        logger.error("Veuillez d'abord exécuter votre script d'ingestion.")
        return

    # 2. Ouverture de la base et extraction des données
    logger.info(f"📖 Chargement de la base Chroma depuis {CHROMA_PATH}...")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    # On récupère les embeddings et les documents déjà stockés
    # C'est ici qu'on gagne tout le temps de calcul !
    logger.info("Extraction des vecteurs et des textes...")
    data = vectorstore.get(include=['embeddings', 'documents'])
    
    doc_embeddings = data['embeddings']
    documents = data['documents']

    if doc_embeddings is None or len(doc_embeddings) == 0:
        logger.warning("⚠️ La base de données est vide.")
        return

    logger.info(f"✅ {len(documents)} documents chargés et prêts pour l'analyse.")

    logger.info("\n--- 3. Définition des Requêtes ---")
    queries = [
        "Qui sont les personnages principaux du Seigneur des Anneaux ?",
        "Qui du film ou du roman du Seigneur des Anneaux est sorti en premier ?",
        "De quand date la dernière sortie d'un film du Seigneur des Anneaux ?",
        "Qui sont les principaux contributeurs à l'univers du Seigneur des Anneaux ?",
        "Peut-on considérer le Seigneur des Anneaux comme une œuvre de fantasy épique ?"
    ]

    logger.info("\n--- 4. Évaluation : Cosinus vs MMR ---")
    K = 5 
    LAMBDA_MULT = 0.5 

    for i, query in enumerate(queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Requête {i} : {query}")
        logger.info(f"{'='*60}")

        # On transforme la requête en vecteur (seule opération d'embedding nécessaire)
        query_embedding = embedding_model.embed_query(query)

        # ---------------------------------------------------------
        # A. Recherche Classique : Similarité Cosinus
        # ---------------------------------------------------------
        logger.info(f"\n🎯 Top {K} - Similarité Cosinus (Pertinence pure) :")
        top_indices = top_k_similar_indices(query_embedding, doc_embeddings, k=K)
        
        for rank, idx in enumerate(top_indices, 1):
            # Comme doc_embeddings et documents viennent du même .get(), les index correspondent
            apercu = documents[idx][:150].replace('\n', ' ')
            logger.info(f"  {rank}. [Doc {idx}] {apercu}...")

        # ---------------------------------------------------------
        # B. Recherche Diversifiée : MMR
        # ---------------------------------------------------------
        logger.info(f"\n🔀 Top {K} - MMR (Pertinence + Diversité) :")
        mmr_results = mmr_from_documents(
            documents=documents,
            doc_embeddings=doc_embeddings,
            query_embedding=query_embedding,
            k=K,
            lambda_mult=LAMBDA_MULT
        )
        
        for rank, doc_text in enumerate(mmr_results, 1):
            apercu = doc_text[:150].replace('\n', ' ')
            logger.info(f"  {rank}. {apercu}...")

if __name__ == "__main__":
    main()