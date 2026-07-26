"""
agent.py
Agente de preguntas y respuestas sobre el documento de Clínica Vitalis.
"""
import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Cargar variables de entorno desde el archivo .env
load_dotenv()

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")

# ──────────────────────────────────────────────────────────────────────────
# Prompt para "reescribir" preguntas de seguimiento como preguntas autónomas
# ──────────────────────────────────────────────────────────────────────────
CONTEXTUALIZE_PROMPT = """Eres un asistente especializado en reescribir consultas de búsqueda.
Dada una conversación previa y la última pregunta del usuario, tu ÚNICA tarea es devolver una pregunta autónoma y clara en español para buscar en una base de datos vectorial.

REGLAS CRÍTICAS:
- Si el usuario pregunta por un NUEVO servicio, examen o tema (ej. pasa de hablar de 'TAC' a 'prueba de embarazo'), NO mezcles ni incluyas el servicio anterior. Limítate a pedir la información sobre el nuevo servicio (ej. '¿Cuál es el precio y requisitos de la prueba de embarazo?').
- Si la pregunta usa pronombres de seguimiento que dependen del tema anterior (ej. "¿cuánto cuesta?", "¿requiere ayuno?", "¿y para niños?"), reemplaza los pronombres por el servicio específico mencionado justo antes.
- Si la pregunta es un saludo, despedida o ya es clara por sí sola, devuélvela exactamente igual.
- NUNCA respondas la pregunta, no agregues explicaciones, ni uses formato Markdown. Devuelve solo la pregunta reformulada en una sola línea."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

SYSTEM_PROMPT_TEMPLATE = """Eres el asistente virtual oficial de Clínica Vitalis. Tu rol es exclusivamente brindar información administrativa y de servicios basados ÚNICAMENTE en el contexto provisto.

--- SEGURIDAD Y EMERGENCIAS (MÁXIMA PRIORIDAD) ---
1. EMERGENCIAS MÉDICAS: Si el usuario menciona síntomas graves, urgencias o emergencias (ej. dolor torácico, dificultad para respirar, sangrado severo, pérdida de conocimiento), ignora el contexto y responde de inmediato: "Si experimentas una emergencia médica, por favor acude inmediatamente al área de urgencias más cercana o llama al número local de emergencias (911)."
2. DIAGNÓSTICOS Y CONSEJOS MÉDICOS: Queda strictly PROHIBIDO dar diagnósticos, interpretar síntomas o recomendar tratamientos/medicamentos. Aclara que solo proporcionas información general de la clínica.

--- REGLAS DE BÚSQUEDA Y VERACIDAD ---
1. Usa EXCLUSIVAMENTE la información contenida dentro de las etiquetas <contexto></contexto>.
2. Si el servicio, examen, consulta o estudio NO aparece mencionado en el contexto, responde exactamente: "No disponemos de ese servicio en nuestra documentación actual." NO uses conocimientos previos ni inventes precios o servicios.
3. Si el contexto incluye precios o detalles de un servicio, confirma que la clínica SÍ lo ofrece y da la información exacta.
4. Para preguntas ajenas a la clínica (deportes, clima, cultura general), responde: "Solo puedo ayudarte con información sobre los servicios y atención de Clínica Vitalis."

--- ESTILO Y FORMATO ---
1. Sé directo, amable y breve (1 a 2 oraciones máximo).
2. NUNCA uses sintaxis Markdown (sin asteriscos *, sin negritas **, sin hashtags #, sin comillas invertidas `).
3. Escribe montos numéricos de forma sencilla (ejemplo: 500.00 MXN).

<contexto>
{context}
</contexto>"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def cargar_agente(index_path: str = INDEX_PATH):
    # Inicializar embeddings compatibles (FastEmbed usa ONNX Runtime en vez de
    # PyTorch completo, lo que reduce muchísimo el consumo de RAM — clave
    # para correr dentro de los 512 MB del plan Free de Render)
    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    # Configurar el modelo con Groq
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # 1) Retriever consciente del historial
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2) Cadena que combina los documentos recuperados + historial + pregunta
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3) Cadena de recuperación completa
    qa_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    return qa_chain


def _a_texto_plano(texto: str) -> str:
    """Red de seguridad: elimina cualquier símbolo de Markdown que el LLM
    haya podido colar en la respuesta."""
    limpio = texto
    
    # 1. Elimina todos los backticks directamente (adiós al texto verde/código)
    limpio = limpio.replace("`", "")
    
    # 2. Elimina formato de negritas, cursivas, encabezados y listas
    limpio = re.sub(r"\*\*([^*]+)\*\*", r"\1", limpio)        # **negrita**
    limpio = re.sub(r"__([^_]+)__", r"\1", limpio)            # __negrita__
    limpio = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"\1", limpio)  # *cursiva*
    limpio = re.sub(r"(?<!\w)_([^_\n]+)_(?!\w)", r"\1", limpio)    # _cursiva_
    limpio = re.sub(r"^\s{0,3}#{1,6}\s*", "", limpio, flags=re.MULTILINE)  # # encabezados
    limpio = re.sub(r"^\s{0,3}[-*+]\s+", "", limpio, flags=re.MULTILINE)   # - viñetas
    
    return limpio.strip()


def preguntar(qa_chain, pregunta: str, chat_history: list | None = None) -> str:
    """
    Realiza una consulta a la cadena pasándole el historial de conversación opcional.
    """
    if chat_history is None:
        chat_history = []
    resultado = qa_chain.invoke({"input": pregunta, "chat_history": chat_history})
    return _a_texto_plano(resultado["answer"])