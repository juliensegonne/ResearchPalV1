import datetime
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader, WebBaseLoader  #chargement des fichiers
from langchain_text_splitters import RecursiveCharacterTextSplitter  #chunking
from langchain_community.vectorstores import Chroma  #bdd ChromaDB
from langchain_community.embeddings import SentenceTransformerEmbeddings  #embeddings


from tqdm import tqdm

def load_and_chunk(data_dir='data', urls=None):
    documents = []
    date_ingestion = datetime.datetime.now().strftime("%Y-%m-%d")

    # 1. Chargement des fichiers locaux
    loaders_config = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }
    
    print("Chargement des fichiers locaux...")
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
        print("\nChargement des URLs...")
        # On charge les URLs une par une pour voir la progression
        for url in tqdm(urls, desc="Téléchargement Web"):
            try:
                loader_web = WebBaseLoader(url)
                documents.extend(loader_web.load())
            except Exception as e:
                print(f"Erreur sur {url}: {e}")

    # 3. Le Chunking
    if not documents:
        print("Aucun document trouvé.")
        return []

    print(f"\nDécoupage de {len(documents)} documents en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    # Le split est généralement très rapide, mais voici comment voir l'avancée
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Terminé ! {len(chunks)} chunks créés.")
    
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
    
    print(f"{len(chunks)} chunks ont été indexés avec succès dans {path}")
    return vectorstore


# --- LE FLUX FINAL (Ton script principal) ---


"""
#test
data = db.get(limit=3)

print(f"--- Nombre total de chunks : {len(db.get()['ids'])} ---")

for i in range(len(data['documents'])):
    print(f"\n--- Chunk {i+1} ---")
    print(f"Contenu (tronqué) : {data['documents'][i][:100]}...")
    print(f"Métadonnées : {data['metadatas'][i]}")
"""
