import datetime
import logging
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, WebBaseLoader  #chargement des fichiers
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter  #chunking
from langchain_chroma import Chroma  #bdd ChromaDB
from langchain_huggingface import HuggingFaceEmbeddings  #embeddings
from tqdm import tqdm
import re

from utils import load_config

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "chunk_size": 500,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", ". ", " ", ""],
    "embedding_model": "all-mpnet-base-v2",
}

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "indexation_config.json")

_cfg = load_config(_DEFAULTS, _CONFIG_PATH)

CHUNK_SIZE: int = int(_cfg["chunk_size"])
CHUNK_OVERLAP: int = int(_cfg["chunk_overlap"])
SEPARATORS: list[str] = _cfg["separators"]
EMBEDDING_MODEL: str = _cfg["embedding_model"]

_DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug")


def _dump_chunks(chunks: list, filename: str, n: int = 50) -> None:
    """Écrit les *n* premiers chunks dans un fichier texte de debug."""
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    path = os.path.join(_DEBUG_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks[:n]):
            f.write(f"=== Chunk {i + 1} ===")
            f.write(f"\nMétadonnées : {chunk.metadata}\n\n")
            f.write(chunk.page_content)
            f.write("\n\n")
    logger.info(f"🔍 Debug : {min(n, len(chunks))} chunks écrits dans {path}")

def load_and_chunk(data_dir='data', urls=None):
    documents = []
    md_documents = []
    date_ingestion = datetime.datetime.now().strftime("%Y-%m-%d")

    # 1. Chargement des fichiers locaux
    loaders_config = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }
    
    loader_kwargs = {
        ".md": {"mode": "elements"},
    }

    logger.info("Chargement des fichiers locaux...")
    if data_dir:
        for ext, loader_cls in loaders_config.items():
            # Utilisation de show_progress=True (supporté par DirectoryLoader si tqdm est installé)
            loader = DirectoryLoader(
                data_dir, 
                glob=f"**/*{ext}", 
                loader_cls=loader_cls,
                silent_errors=True,
                show_progress=True,
                #loader_kwargs=loader_kwargs.get(ext, {}),
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
                docs = loader_web.load()
                
                # Nettoyage du contenu de chaque document récupéré
                for doc in docs:
                    text = doc.page_content
                    try:
                        text = text.encode('latin-1').decode('utf-8')
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        # Si la conversion échoue, c'est que le texte était déjà correctement encodé
                        pass
                    # 1. Remplace les espaces/tabulations multiples par un seul espace
                    text = re.sub(r'[ \t]+', ' ', text)
                    # 2. Réduit les suites de 3 sauts de ligne (ou plus) à 2 sauts de ligne (pour garder les paragraphes)
                    text = re.sub(r'\n\s*\n', '\n\n', text)
                
                    doc.page_content = text.strip()
                url_documents.extend(docs)
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
        text_chunks = text_splitter.split_documents(documents)
        _dump_chunks(text_chunks, "chunks_texte.txt")
        chunks.extend(text_chunks)

    # Chunking des fichiers Markdown (par headers)
    if md_documents:
        logger.info(f"\nDécoupage de {len(md_documents)} documents Markdown en chunks...")
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        # # Second pass : redécouper les sections trop longues
        # md_text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=CHUNK_SIZE,
        #     chunk_overlap=CHUNK_OVERLAP,
        # )
        md_chunks = []
        for doc in md_documents:
            header_splits = md_header_splitter.split_text(doc.page_content)
            # sub_chunks = md_text_splitter.split_documents(header_splits)
            # for chunk in sub_chunks:
            #     # Fusionner metadata du document original + headers extraits
            #     chunk.metadata = {**doc.metadata, **chunk.metadata}
            for chunk in header_splits:
                chunk.metadata = {**doc.metadata, **chunk.metadata}
            md_chunks.extend(header_splits)
        _dump_chunks(md_chunks, "chunks_markdown.txt")
        chunks.extend(md_chunks)

    # Chunking des pages web (séparateurs HTML)
    if url_documents:
        logger.info(f"\nDécoupage de {len(url_documents)} pages web en chunks...")
        html_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.HTML,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        html_chunks = html_splitter.split_documents(url_documents)
        _dump_chunks(html_chunks, "chunks_web.txt")
        chunks.extend(html_chunks)

    logger.info(f"✅ Terminé ! {len(chunks)} chunks créés.")
    
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
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Création et persistance de la base
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=path
    )
    
    logger.info(f"{len(chunks)} chunks ont été indexés avec succès dans {path}")
    return vectorstore
