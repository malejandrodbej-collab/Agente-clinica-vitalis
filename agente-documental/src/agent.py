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

PROMPT_TEMPLATE = """Eres el asistente virtual de Clínica Vitalis. Respondes preguntas
del usuario ÚNICAMENTE con base en el siguiente contexto extraído de la
documentación interna. Si la respuesta no está en el contexto, indica
claramente que no cuentas con esa información en los documentos disponibles.
No inventes datos, horarios, porcentajes ni montos.

Sé preciso y directo: responde exactamente lo que se pregunta, sin agregar
datos relacionados que no se pidieron. Por ejemplo, si preguntan a qué hora
abre la clínica, contesta solo la hora de apertura, no el horario completo
de todos los días. Si el usuario necesita más detalle, lo puede pedir en una
siguiente pregunta. Evita repetir el contexto completo o citarlo tal cual;
extrae y resume únicamente el dato solicitado en una o dos oraciones.

Contexto:
{context}

Pregunta: {input}

Respuesta directa y breve en español (solo el dato pedido):"""


def cargar_agente(index_path: str = INDEX_PATH):
    # Inicializar embeddings compatibles (multilingüe: el índice y las consultas
    # deben usar EXACTAMENTE el mismo modelo, aquí y en construir_indice.py)
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})

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