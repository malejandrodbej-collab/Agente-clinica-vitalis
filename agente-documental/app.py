"""
app.py
Interfaz web (Streamlit) para el agente documental de Clínica Vitalis.
Envuelve las funciones de src/agent.py en un chat accesible desde el navegador.
"""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from src.agent import cargar_agente, preguntar

# ──────────────────────────────────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clínica Vitalis · Asistente virtual",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Estilos — paleta clínica seria, tipografía con un toque editorial
# (sin cambios respecto a la versión original)
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

    :root {
        --bg: #F6F7F4;
        --surface: #FFFFFF;
        --ink: #1C2B2A;
        --muted: #5C6B66;
        --primary: #2F5D57;
        --primary-dark: #234641;
        --accent: #8FA998;
        --border: #E1E6E1;
        --alert-bg: #FBF3EC;
        --alert-border: #E7C9A9;
    }

    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
        box-shadow: none;
    }
    div[data-testid="stDecoration"] {display: none;}

    .stApp {
        font-family: 'Inter', sans-serif;
        color: var(--ink);
    }

    .vitalis-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 1.1rem 0 0.9rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.4rem;
    }
    .vitalis-mark {
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: var(--primary);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.15rem;
        flex-shrink: 0;
    }
    .vitalis-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.35rem;
        color: var(--ink);
        line-height: 1.15;
    }
    .vitalis-subtitle {
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 0.1rem;
    }

    .pulse-line { width: 100%; height: 22px; margin: 0.2rem 0 1.1rem 0; }
    .pulse-line svg { width: 100%; height: 100%; display: block; }

    div[data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.35rem 0.2rem;
        margin-bottom: 0.6rem;
    }

    .scope-note {
        background: var(--alert-bg);
        border: 1px solid var(--alert-border);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        font-size: 0.82rem;
        color: var(--ink);
        margin-bottom: 1rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] h3 {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        color: var(--ink);
    }
    .sidebar-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
        margin-top: 1.1rem;
        margin-bottom: 0.35rem;
    }

    .stButton button {
        background: var(--surface);
        border: 1px solid var(--border);
        color: var(--ink);
        border-radius: 8px;
        font-size: 0.82rem;
        text-align: left;
        padding: 0.5rem 0.7rem;
        width: 100%;
    }
    .stButton button:hover {
        border-color: var(--primary);
        color: var(--primary);
    }

    div[data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif;
    }
    </style>

    <div class="vitalis-header">
        <div class="vitalis-mark">CV</div>
        <div>
            <div class="vitalis-title">Clínica Vitalis</div>
            <div class="vitalis-subtitle">Asistente virtual documental</div>
        </div>
    </div>
    <div class="pulse-line">
        <svg viewBox="0 0 600 22" preserveAspectRatio="none">
            <polyline points="0,11 220,11 240,3 255,19 270,11 290,11 300,11 310,4 320,18 330,11 350,11 600,11"
                      fill="none" stroke="#8FA998" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
        </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="scope-note">
        Este asistente responde con base en la documentación interna de Clínica Vitalis
        (política de privacidad, turnos, cancelaciones, convenios e instrucciones
        pre/post consulta). No sustituye una valoración médica profesional.
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────
# Carga del agente (una sola vez por sesión de servidor)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def obtener_agente():
    return cargar_agente()


try:
    with st.spinner("Preparando el asistente…"):
        qa_chain = obtener_agente()
    agente_disponible = True
except Exception as e:
    agente_disponible = False
    st.error(
        "No fue posible inicializar el asistente. Verifica que el índice "
        "vectorial (`vector_store/`) exista y que las variables de entorno "
        f"necesarias estén configuradas.\n\nDetalle técnico: `{e}`"
    )

# ──────────────────────────────────────────────────────────────────────────
# Sidebar — contexto y preguntas de ejemplo
# ──────────────────────────────────────────────────────────────────────────
EJEMPLOS = [
    "¿Cuánto tiempo antes debo llegar a mi cita?",
    "¿Qué pasa si cancelo con menos de 24 horas?",
    "¿Qué cobertura tiene Seguros del Valle?",
    "¿Necesito ayuno antes de mi consulta?",
]

with st.sidebar:
    st.markdown("### Sobre este asistente")
    st.markdown(
        "Responde preguntas frecuentes de pacientes usando únicamente "
        "la documentación interna de la clínica. Si algo no está en los "
        "documentos, el asistente lo indicará en lugar de inventar una respuesta."
    )
    st.markdown('<div class="sidebar-label">Preguntas de ejemplo</div>', unsafe_allow_html=True)
    for pregunta in EJEMPLOS:
        if st.button(pregunta, key=f"ejemplo_{pregunta}", disabled=not agente_disponible):
            st.session_state["pregunta_pendiente"] = pregunta
    st.markdown('<div class="sidebar-label">Nota</div>', unsafe_allow_html=True)
    st.caption("Para urgencias médicas, contacta directamente a la clínica o a servicios de emergencia.")

    if st.button("🗑️ Reiniciar conversación", disabled=not agente_disponible):
        st.session_state["mensajes"] = [
            {
                "role": "assistant",
                "content": "Hola, soy el asistente virtual de Clínica Vitalis. ¿En qué puedo ayudarte hoy?",
            }
        ]
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────
# Historial de chat (para mostrar en pantalla)
# ──────────────────────────────────────────────────────────────────────────
if "mensajes" not in st.session_state:
    st.session_state["mensajes"] = [
        {
            "role": "assistant",
            "content": "Hola, soy el asistente virtual de Clínica Vitalis. ¿En qué puedo ayudarte hoy?",
        }
    ]

for mensaje in st.session_state["mensajes"]:
    avatar = "🩺" if mensaje["role"] == "assistant" else "🧑"
    with st.chat_message(mensaje["role"], avatar=avatar):
        st.markdown(mensaje["content"])

# ──────────────────────────────────────────────────────────────────────────
# Conversión del historial mostrado en pantalla al formato que espera
# LangChain (HumanMessage / AIMessage), para el history-aware retriever.
# Se limita a los últimos N turnos para no inflar el contexto sin necesidad.
# ──────────────────────────────────────────────────────────────────────────
MAX_TURNOS_HISTORIAL = 6  # ~6 pares pregunta/respuesta hacia atrás


def construir_historial_langchain(mensajes: list) -> list:
    historial = []
    for m in mensajes:
        if m["role"] == "user":
            historial.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            historial.append(AIMessage(content=m["content"]))
    # nos quedamos solo con los últimos N*2 mensajes (N turnos completos)
    return historial[-(MAX_TURNOS_HISTORIAL * 2):]


# Pregunta disparada desde el sidebar (si la hay) o desde el campo de texto
pregunta_usuario = st.session_state.pop("pregunta_pendiente", None) or st.chat_input(
    "Escribe tu pregunta…", disabled=not agente_disponible
)

if pregunta_usuario:
    # Importante: el historial que se le pasa al agente se construye ANTES
    # de agregar la pregunta actual a session_state, para que no se duplique
    # (la pregunta actual va aparte, como "input").
    historial_previo = construir_historial_langchain(st.session_state["mensajes"])

    st.session_state["mensajes"].append({"role": "user", "content": pregunta_usuario})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(pregunta_usuario)

    with st.chat_message("assistant", avatar="🩺"):
        with st.spinner("Consultando la documentación…"):
            try:
                respuesta = preguntar(qa_chain, pregunta_usuario, historial_previo)
            except Exception as e:
                respuesta = (
                    "Ocurrió un problema al procesar tu pregunta. "
                    f"Detalle técnico: `{e}`"
                )
        st.markdown(respuesta)

    st.session_state["mensajes"].append({"role": "assistant", "content": respuesta})
