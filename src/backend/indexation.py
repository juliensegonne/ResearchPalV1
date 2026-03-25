import datetime
import logging
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, WebBaseLoader  #chargement des fichiers
from langchain_text_splitters import RecursiveCharacterTextSplitter  #chunking
from langchain_chroma import Chroma  #bdd ChromaDB
from langchain_huggingface import HuggingFaceEmbeddings  #embeddings


from tqdm import tqdm

logger = logging.getLogger("uvicorn.error")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Ordre de préférence pour le split

def load_and_chunk(data_dir='data', urls=None):
    documents = []
    date_ingestion = datetime.datetime.now().strftime("%Y-%m-%d")

    # 1. Chargement des fichiers locaux
    loaders_config = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }
    
    logger.info("Chargement des fichiers locaux...")
    for ext, loader_cls in loaders_config.items():
        # Utilisation de show_progress=True (supporté par DirectoryLoader si tqdm est installé)
        loader = DirectoryLoader(
            data_dir, 
            glob=f"**/*{ext}", 
            loader_cls=loader_cls,
            silent_errors=True,
            show_progress=True # LangChain affichera une barre interne
        )
        
        batch = loader.load()
        
        # On peut aussi ajouter une barre manuelle pour le nettoyage des métadonnées
        for d in tqdm(batch, desc=f"Nettoyage {ext}", leave=False):
            original_source = d.metadata.get("source", "inconnue")
            d.metadata = {
                "source": original_source,
                "doc_type": "pdf" if ext == ".pdf" else "document_texte",
                "ingestion_date": date_ingestion
            }
        documents.extend(batch)

    # 2. Chargement des URLs
    if urls:
        logger.info("\nChargement des URLs...")
        # On charge les URLs une par une pour voir la progression
        for url in tqdm(urls, desc="Téléchargement Web"):
            try:
                loader_web = WebBaseLoader(url)
                documents.extend(loader_web.load())
            except Exception as e:
                logger.exception("Erreur sur %s: %s", url, e)

    # 3. Le Chunking
    if not documents:
        logger.warning("Aucun document trouvé.")
        return []

    logger.info(f"\nDécoupage de {len(documents)} documents en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )

    # Le split est généralement très rapide, mais voici comment voir l'avancée
    chunks = text_splitter.split_documents(documents)
    logger.info(f"✅ Terminé ! {len(chunks)} chunks créés.")
    logger.info(f"Exemple de chunk : {chunks[0].page_content}... | Métadonnées : {chunks[0].metadata}")
    
    return chunks

"""
# --- Exemple d'utilisation ---
mes_urls = ["https://fr.wikipedia.org/wiki/Intelligence_artificielle"]
tous_mes_chunks = load_and_chunk(data_dir='data', urls=mes_urls)

print(f"Nombre de chunks créés : {len(tous_mes_chunks)}")
if tous_mes_chunks:
    print(f"Exemple de métadonnées : {tous_mes_chunks[0].metadata}")"""
    
def store_in_chroma(chunks, path="./chroma_db"):
    """
    Prend une liste de chunks et les stocke dans une base ChromaDB.
    """
    # On définit le modèle d'embedding
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    
    # Création et persistance de la base
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=path
    )
    
    logger.info(f"{len(chunks)} chunks ont été indexés avec succès dans {path}")
    return vectorstore

#chunks = load_and_chunk(data_dir='data', urls=["https://fr.wikipedia.org/wiki/Masashi_Kishimoto"]) 
#store_in_chroma(chunks, path="./chroma_db") 