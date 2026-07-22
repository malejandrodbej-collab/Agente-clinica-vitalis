"""
debug_retriever.py
Script para inspeccionar qué fragmentos recupera el vector store
para una pregunta dada, sin pasar por el modelo de lenguaje.

Uso:
    python debug_retriever.py
"""
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Ajusta esta ruta si tu vector_store está en otro lugar
INDEX_PATH = os.path.join(os.path.dirname(__file__), "vector_store")

PREGUNTAS_DE_PRUEBA = [
    "¿Qué servicios de laboratoriales ofrecen?",
    "¿Qué precios tienen los exámenes laboratoriales?",
    "¿Cuánto cuesta la biometría hemática?",
]


def main():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

    for pregunta in PREGUNTAS_DE_PRUEBA:
        print("=" * 80)
        print(f"PREGUNTA: {pregunta}")
        print("=" * 80)
        docs = retriever.invoke(pregunta)
        for i, doc in enumerate(docs, 1):
            print(f"\n--- Fragmento {i} ---")
            print(doc.page_content[:400])
        print("\n")


if __name__ == "__main__":
    main()
