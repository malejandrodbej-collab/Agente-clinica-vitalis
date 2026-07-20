"""
agent.py
Agente de preguntas y respuestas sobre el documento de Clínica Vitalis.

Flujo:
  1. Carga el índice vectorial construido por ingest.py.
  2. Ante una pregunta, recupera los fragmentos del documento más
     relevantes (retrieval).
  3. Envía esos fragmentos + la pregunta al modelo de lenguaje, que
     genera una respuesta en lenguaje natural basada únicamente en
     ese contexto (generation) -> de ahí "RAG": Retrieval-Augmented
     Generation.

Ejecución interactiva por consola:
    python src/agent.py
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")

PROMPT_TEMPLATE = """Eres el asistente virtual de Clínica Vitalis. Responde la pregunta
del usuario ÚNICAMENTE con base en el siguiente contexto extraído de la
documentación interna. Si la respuesta no está en el contexto, indica
claramente que no cuentas con esa información en los documentos disponibles.
No inventes datos, horarios, porcentajes ni montos.

Contexto:
{context}

Pregunta: {question}

Respuesta clara y concisa en español:"""


def cargar_agente(index_path: str = INDEX_PATH):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain


def preguntar(qa_chain, pregunta: str) -> str:
    resultado = qa_chain.invoke({"query": pregunta})
    return resultado["result"]


def main():
    print("Cargando agente de Clínica Vitalis... (Ctrl+C para salir)\n")
    qa_chain = cargar_agente()

    while True:
        try:
            pregunta = input("Pregunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego.")
            break

        if not pregunta:
            continue

        respuesta = preguntar(qa_chain, pregunta)
        print(f"Respuesta: {respuesta}\n")


if __name__ == "__main__":
    main()
