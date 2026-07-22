"""
construir_indice.py
Construye (o reconstruye) el índice vectorial FAISS a partir del PDF
de documentación de Clínica Vitalis, ubicado en la carpeta "data".

Debe vivir en la misma carpeta que agent.py, porque ambos calculan
la ruta de vector_store de forma relativa a su propia ubicación.
"""
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Carpeta "data", un nivel arriba de este archivo (mismo patrón que vector_store)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Debe coincidir EXACTAMENTE con el INDEX_PATH de agent.py
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")


def encontrar_pdf(data_dir: str) -> str:
    """Busca el primer PDF dentro de la carpeta data."""
    pdfs = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No encontré ningún .pdf dentro de: {os.path.abspath(data_dir)}"
        )
    if len(pdfs) > 1:
        print(f"Aviso: hay {len(pdfs)} PDFs en data/, voy a usar: {os.path.basename(pdfs[0])}")
    return pdfs[0]


def construir_indice():
    pdf_path = encontrar_pdf(DATA_DIR)
    print(f"Cargando PDF desde: {os.path.abspath(pdf_path)}")

    loader = PyPDFLoader(pdf_path)
    documentos = loader.load()
    print(f"PDF cargado: {len(documentos)} página(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documentos)
    print(f"Documento dividido en {len(chunks)} fragmentos.")

    print("Generando embeddings (puede tardar un poco la primera vez)...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_PATH)
    print(f"Índice guardado en: {os.path.abspath(INDEX_PATH)}")


if __name__ == "__main__":
    construir_indice()
