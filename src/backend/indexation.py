import datetime
import json
import logging
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, UnstructuredMarkdownLoader, WebBaseLoader  #chargement des fichiers
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language  #chunking
from langchain_chroma import Chroma  #bdd ChromaDB
from langchain_huggingface import HuggingFaceEmbeddings  #embeddings


from tqdm import tqdm

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", ". ", " ", ""],
}

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "indexation_config.json")


def _load_config() -> dict:
    cfg = dict(_DEFAULTS)
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH) as f:
                cfg.update(json.load(f))
            logger.info(f"⚙️ Config indexation chargée depuis {_CONFIG_PATH}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur lecture config indexation, défauts utilisés : {e}")
    return cfg


_cfg = _load_config()

CHUNK_SIZE: int = int(_cfg["chunk_size"])
CHUNK_OVERLAP: int = int(_cfg["chunk_overlap"])
SEPARATORS: list[str] = _cfg["separators"]

def load_and_chunk(data_dir='data', urls=None):
    documents = []
    md_documents = []
    date_ingestion = datetime.datetime.now().strftime("%Y-%m-%d")

    # 1. Chargement des fichiers locaux
    loaders_config = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    
    loader_kwargs = {
        ".md": {"mode": "elements"},
    }

    logger.info("Chargement des fichiers locaux...")
    for ext, loader_cls in loaders_config.items():
        # Utilisation de show_progress=True (supporté par DirectoryLoader si tqdm est installé)
        loader = DirectoryLoader(
            data_dir, 
            glob=f"**/*{ext}", 
            loader_cls=loader_cls,
            silent_errors=True,
            show_progress=True,
            loader_kwargs=loader_kwargs.get(ext, {}),
        )
        
        batch = loader.load()
        
        # On peut aussi ajouter une barre manuelle pour le nettoyage des métadonnées
        for d in tqdm(batch, desc=f"Nettoyage {ext}", leave=False):
            original_source = d.metadata.get("source", "inconnue")
            d.metadata = {
                "source": original_source,
                "doc_type": {  ".pdf": "pdf", ".txt": "texte", ".md": "markdown" }[ext],
                "ingestion_date": date_ingestion
            }
        if ext == ".md":
            md_documents.extend(batch)
        else:
            documents.extend(batch)

    # 2. Chargement des URLs
    url_documents = []
    if urls:
        logger.info("\nChargement des URLs...")
        for url in tqdm(urls, desc="Téléchargement Web"):
            try:
                loader_web = WebBaseLoader(url)
                url_documents.extend(loader_web.load())
            except Exception as e:
                logger.exception("Erreur sur %s: %s", url, e)

        for d in url_documents:
            d.metadata = {
                "source": d.metadata.get("source", "inconnue"),
                "doc_type": "web",
                "ingestion_date": date_ingestion
            }

    # 3. Le Chunking
    if not documents and not md_documents and not url_documents:
        logger.warning("Aucun document trouvé.")
        return []

    chunks = []

    # Chunking des fichiers locaux PDF/TXT (séparateurs texte)
    if documents:
        logger.info(f"\nDécoupage de {len(documents)} documents texte en chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )
        chunks.extend(text_splitter.split_documents(documents))

    # Chunking des fichiers Markdown (séparateurs Markdown)
    if md_documents:
        logger.info(f"\nDécoupage de {len(md_documents)} documents Markdown en chunks...")
        md_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks.extend(md_splitter.split_documents(md_documents))

    # Chunking des pages web (séparateurs HTML)
    if url_documents:
        logger.info(f"\nDécoupage de {len(url_documents)} pages web en chunks...")
        html_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.HTML,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks.extend(html_splitter.split_documents(url_documents))

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