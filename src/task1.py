import PyPDF2
import requests
import os

def read_pdf(file) :
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_txt(file) :
    with open(file, 'r') as f:
        text = f.read()
    return text

def read_url(url) :
    response = requests.get(url)
    return response.text


# réfléchir pour les url dans cette fonction
def load_data_texts(data_dir='data'):
    """
    Parcourt le dossier `data_dir`, lit les fichiers supportés et renvoie une liste de textes.
    Supporte : .pdf, .txt, .md, .html, .htm
    Retour : list de strings (texte de chaque document). En cas d'erreur, une string décrivant l'erreur est ajoutée à la liste.
    """

    texts = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext == '.pdf':
                    # read_pdf attend un fichier (file-like), on ouvre en binaire
                    with open(path, 'rb') as f:
                        texts.append(read_pdf(f))
                elif ext in ('.txt', '.md', '.html', '.htm'):
                    # read_txt accepte un chemin vers le fichier
                    texts.append(read_txt(path))
                else:
                    # ignorer les fichiers non supportés
                    continue
            except Exception as e:
                texts.append(f'ERROR_READING_FILE {path}: {e}')

    return texts

texts = load_data_texts()
print(f"Loaded {len(texts)} documents.")
print(f"First document preview:\n{texts[0][:500]}")
