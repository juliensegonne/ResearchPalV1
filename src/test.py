import datetime
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, WebBaseLoader  #chargement des fichiers
from langchain_text_splitters import RecursiveCharacterTextSplitter  #chunking
from langchain_community.vectorstores import Chroma  #bdd ChromaDB
from langchain_community.embeddings import SentenceTransformerEmbeddings  #embeddings


def load_and_chunk(data_dir='data', urls=None):
    """
    Charge les fichiers d'un dossier et les URLs, puis les découpe en chunks.
    """
    documents = []
    date_ingestion = datetime.datetime.now().strftime("%Y-%m-%d")

    # 1. Chargement des fichiers locaux (.pdf, .txt, .md)
    # DirectoryLoader gère le parcours du dossier
    # On définit quel loader utiliser pour chaque extension
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }
    
    for ext, loader_cls in loaders.items():
        loader = DirectoryLoader(
            data_dir, 
            glob=f"**/*{ext}", 
            loader_cls=loader_cls,
            silent_errors=True # Évite de crash si un fichier est corrompu
        )
        batch = loader.load()
        for d in batch:
            # --- NETTOYAGE DES MÉTADONNÉES ---
            # On récupère la source (chemin du fichier) que LangChain a déjà extraite
            original_source = d.metadata.get("source", "inconnue")
            
            # On ÉCRASE complètement le dictionnaire metadata pour ne garder que tes 3 champs
            d.metadata = {
                "source": original_source,
                "doc_type": "pdf" if ext == ".pdf" else "document_texte",
                "ingestion_date": date_ingestion
            }
        
        documents.extend(batch)

    # 2. Chargement des URLs
    if urls:
        # WebBaseLoader extrait le texte propre d'une page HTML
        loader_web = WebBaseLoader(urls)
        documents.extend(loader_web.load())

    # 3. Le Chunking (Version Document)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    # split_documents est magique : il découpe le texte 
    # MAIS il garde les métadonnées (source) pour chaque chunk !
    chunks = text_splitter.split_documents(documents)
    
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
    # On définit le modèle d'embedding (la "traduction" en vecteurs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Création et persistance de la base
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=path
    )
    
    print(f"✅ {len(chunks)} chunks ont été indexés avec succès dans {path}")
    return vectorstore


# --- LE FLUX FINAL (Ton script principal) ---

# Étape 1 : On prépare les données
chunks = load_and_chunk(data_dir='data', urls=["https://google.com"])

# Étape 2 : On les stocke
db = store_in_chroma(chunks)

"""
#test
data = db.get(limit=3)

print(f"--- Nombre total de chunks : {len(db.get()['ids'])} ---")

for i in range(len(data['documents'])):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Contenu (tronqué) : {data['documents'][i][:100]}...")
    print(f"Métadonnées : {data['metadatas'][i]}")
"""
