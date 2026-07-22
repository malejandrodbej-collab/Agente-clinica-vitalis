"""
agent.py
Agente de preguntas y respuestas sobre el documento de Clínica Vitalis.
"""
import os
import re
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

Responde siempre en texto plano, sin usar Markdown: no uses backticks,
asteriscos, guiones para listas, encabezados con '#', ni ningún otro
símbolo de formato. Escribe los montos y datos exactamente como texto
normal (ejemplo: 500.00 MXN), nunca entre comillas invertidas ni con
énfasis especial.

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


def _a_texto_plano(texto: str) -> str:
    """Red de seguridad: elimina cualquier símbolo de Markdown que el LLM
    haya podido colar en la respuesta (backticks, negritas, cursivas,
    encabezados, viñetas), para que todo se muestre como texto plano y
    del mismo color en la interfaz."""
    limpio = texto
    limpio = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", limpio)   # `código` o ```bloque```
    limpio = re.sub(r"\*\*([^*]+)\*\*", r"\1", limpio)        # **negrita**
    limpio = re.sub(r"__([^_]+)__", r"\1", limpio)            # __negrita__
    limpio = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", limpio)  # *cursiva*
    limpio = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", limpio)    # _cursiva_
    limpio = re.sub(r"^\s{0,3}#{1,6}\s*", "", limpio, flags=re.MULTILINE)  # # encabezados
    limpio = re.sub(r"^\s{0,3}[-*+]\s+", "", limpio, flags=re.MULTILINE)   # - viñetas
    return limpio.strip()


def preguntar(qa_chain, pregunta: str) -> str:
    resultado = qa_chain.invoke({"input": pregunta})
    return _a_texto_plano(resultado["answer"])