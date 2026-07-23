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
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Cargar variables de entorno desde el archivo .env
load_dotenv()

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")

# ──────────────────────────────────────────────────────────────────────────
# Prompt para "reescribir" preguntas de seguimiento como preguntas autónomas
# ANTES de buscar en el índice vectorial. Esto es lo que faltaba: sin esto,
# una pregunta como "en orina?" se busca tal cual, aislada del historial,
# y el retriever no tiene forma de saber que se refiere a "prueba de
# embarazo en orina".
# ──────────────────────────────────────────────────────────────────────────
CONTEXTUALIZE_PROMPT = """Dado el historial de la conversación y la última pregunta del usuario,
reformula la consulta para que sea una pregunta completa, autónoma e independiente,
entendible por sí sola sin necesidad de leer la charla previa.

Instrucciones:
1. Si la pregunta depende del contexto anterior (ej. "¿cuánto cuesta?", "¿y en orina?",
   "¿tienen?"), sustituye los pronombres o referentes explícitos por el servicio o tema exacto
   al que se refieren.
2. Si la pregunta ya es clara y autónoma por sí misma, devuélvela exactamente igual.
3. NO respondas la pregunta bajo ninguna circunstancia. Solo devuélvela reformulada en una
   sola línea."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

SYSTEM_PROMPT_TEMPLATE = """Eres el asistente virtual de Clínica Vitalis. Tu objetivo es responder
preguntas de los usuarios ÚNICAMENTE con base en el contexto proporcionado.

REGLAS DE BÚSQUEDA E INFERENCIA:
- Responde únicamente con la información presente en el contexto adjunto.
- Si en el contexto se incluyen precios, tarifas, requisitos o indicaciones sobre un examen,
  prueba o servicio, asume y confirma que la clínica SÍ ofrece dicho servicio.
- Si la información no aparece ni se deduce explícitamente del contexto, indica claramente
  que no cuentas con esos datos en la documentación disponible. No inventes montos, horarios ni políticas.

REGLAS DE ESTILO Y CONCISIÓN:
- Sé directo y breve: responde en 1 o 2 oraciones exactamente lo que se pregunta, sin agregar datos no solicitados.
- Si preguntan la hora de apertura, da solo la hora de apertura, no el horario de toda la semana.

REGLAS STRICTAS DE FORMATO (TEXTO PLANO):
- Escribe ÚNICAMENTE en texto plano.
- Queda PROHIBIDO usar sintaxis Markdown: NO uses comillas invertidas (backticks `), asteriscos (*),
  almohadillas (#), ni guiones de lista (-).
- Escribe cifras y monedas como texto continuo común (ejemplo correcto: 500.00 MXN | ejemplo incorrecto: `500.00 MXN`)."""

Contexto:
{context}

Respuesta directa y breve en español (solo el dato pedido):"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_TEMPLATE),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def cargar_agente(index_path: str = INDEX_PATH):
    # Inicializar embeddings compatibles (multilingüe: el índice y las consultas
    # deben usar EXACTAMENTE el mismo modelo, aquí y en construir_indice.py)
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    # k=8 en lugar de 6: da un poco más de margen de recall para preguntas
    # cuyo phrasing no calza tan directo con el texto del documento (p. ej.
    # "servicios de análisis laboratoriales" vs. cómo esté redactado en el PDF).
    # Si sigue fallando en temas específicos, el siguiente paso es revisar
    # construir_indice.py: tamaño de chunk / overlap.
    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    # Configurar el modelo con Groq
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # 1) Retriever "consciente del historial": reformula la pregunta de
    #    seguimiento en una pregunta autónoma antes de buscar en el índice.
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 2) Cadena que combina los documentos recuperados + historial + pregunta
    #    para generar la respuesta final.
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3) Cadena de recuperación completa, ahora con el retriever consciente
    #    del historial en vez del retriever "pelado" original.
    qa_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

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


def preguntar(qa_chain, pregunta: str, chat_history: list | None = None) -> str:
    """
    chat_history: lista de mensajes langchain_core (HumanMessage/AIMessage)
    con los turnos previos de la conversación. Si no se pasa nada, se
    comporta como antes (sin memoria), pero ya no es lo recomendado desde
    app.py — ver construir_historial() allá.
    """
    if chat_history is None:
        chat_history = []
    resultado = qa_chain.invoke({"input": pregunta, "chat_history": chat_history})
    return _a_texto_plano(resultado["answer"])
