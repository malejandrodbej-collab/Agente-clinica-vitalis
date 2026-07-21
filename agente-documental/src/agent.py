"""
agent.py
Agente de preguntas y respuestas sobre el documento de Clínica Vitalis.
"""
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# Cargar variables de entorno desde el archivo .env
load_dotenv()

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")

PROMPT_TEMPLATE = """Eres el asistente virtual de Clínica Vitalis. Responde la pregunta
del usuario ÚNICAMENTE con base en el siguiente contexto extraído de la
documentación interna. Si la respuesta no está en el contexto, indica
claramente que no cuentas con esa información en los documentos disponibles.
No inventes datos, horarios, porcentajes ni montos.

Contexto:
{context}

Pregunta: {input}

Respuesta clara y concisa en español:"""


def cargar_agente(index_path: str = INDEX_PATH):
    # Inicializar embeddings compatibles
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Configurar el modelo con Groq
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Configurar el prompt
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Crear cadena de combinación de documentos
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # Crear cadena de recuperación
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return qa_chain


def preguntar(qa_chain, pregunta: str) -> str:
    resultado = qa_chain.invoke({"input": pregunta})
    return resultado["answer"]