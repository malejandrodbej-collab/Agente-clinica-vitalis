"""
ingest.py
Lee el documento fuente (PDF), lo divide en fragmentos manejables
y construye un índice vectorial (FAISS) para búsqueda semántica.

Este paso se ejecuta una sola vez (o cada vez que cambia el documento
fuente) y guarda el índice en disco para que el agente lo reutilice
sin tener que re-procesar el PDF en cada consulta.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "clinica_vitalis_documentacion.pdf")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")


def cargar_documento(path: str = DATA_PATH):
    """Carga el PDF y devuelve una lista de objetos Document (uno por página)."""
    loader = PyPDFLoader(path)
    paginas = loader.load()
    print(f"Documento cargado: {len(paginas)} páginas.")
    return paginas


def dividir_en_fragmentos(paginas, chunk_size: int = 800, chunk_overlap: int = 150):
    """
    Divide el documento en fragmentos (chunks) más pequeños.
    chunk_overlap evita perder contexto en los bordes de cada fragmento,
    lo cual es importante para políticas y tablas que se extienden entre secciones.
    El separador "\n¿" tiene prioridad para que, en las secciones de preguntas
    frecuentes, cada pregunta con su respuesta tienda a quedar en su propio
    fragmento en vez de mezclarse con las preguntas vecinas.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n¿", "\n\n", "\n", ". ", " ", ""],
    )
    fragmentos = splitter.split_documents(paginas)
    print(f"Documento dividido en {len(fragmentos)} fragmentos.")
    return fragmentos


def construir_indice(fragmentos, index_path: str = INDEX_PATH):
    """Genera embeddings para cada fragmento y construye/guarda el índice FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(fragmentos, embeddings)
    vector_store.save_local(index_path)
    print(f"Índice vectorial guardado en: {index_path}")
    return vector_store


def main():
    paginas = cargar_documento()
    fragmentos = dividir_en_fragmentos(paginas)
    construir_indice(fragmentos)


if __name__ == "__main__":
    main()
