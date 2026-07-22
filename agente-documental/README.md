# Agente Documental — Clínica Vitalis

Agente de inteligencia artificial que responde preguntas en lenguaje natural
sobre la documentación interna de una clínica de salud (política de
privacidad, FAQ de turnos, cancelaciones, convenios y coberturas, e
instrucciones pre/post consulta), sin que el usuario tenga que abrir ni
buscar dentro de los documentos.

Proyecto desarrollado para el Challenge "Agente Alura" del curso de
Oracle AI.

## Descripción general

Cualquier persona colaboradora o paciente puede preguntarle al agente,
por ejemplo, "¿cuánto tiempo antes debo llegar a mi cita?" o "¿qué
cobertura tiene Seguros del Valle en especialidades?", y recibir una
respuesta directa extraída del documento fuente, sin tener que leerlo
completo.

El documento fuente utilizado (`data/clinica_vitalis_documentacion.pdf`)
consolida cinco políticas de una clínica de salud ficticia, elegido por
su relación con mi formación en Tecnologías Biomédicas.

## Arquitectura de la solución

El agente sigue un patrón **RAG (Retrieval-Augmented Generation)**:

```
                    ┌───────────────────────┐
                    │   PDF fuente           │
                    │ (clínica_vitalis.pdf)  │
                    └───────────┬───────────┘
                                │  1. Carga (PyPDFLoader)
                                ▼
                    ┌───────────────────────┐
                    │  División en           │
                    │  fragmentos (chunks)   │
                    └───────────┬───────────┘
                                │  2. Embeddings (HuggingFace, local)
                                ▼
                    ┌───────────────────────┐
                    │  Índice vectorial      │
                    │  (FAISS)               │
                    └───────────┬───────────┘
                                │  3. Retrieval (top-k similares)
   Pregunta del  ──────────────▶│
   usuario                      ▼
                    ┌───────────────────────┐
                    │  LLM (Groq · Llama 3.1)│
                    │  + prompt con contexto │
                    └───────────┬───────────┘
                                │  4. Generación
                                ▼
                        Respuesta en
                        lenguaje natural
```

1. **Ingesta** (`src/ingest.py`): lee el PDF, lo divide en fragmentos con
   solapamiento (para no perder contexto entre secciones) y genera un
   índice vectorial FAISS guardado en disco.
2. **Agente** (`src/agent.py`): ante cada pregunta, recupera los
   fragmentos más relevantes del índice y se los pasa junto con la
   pregunta a un modelo de lenguaje, que responde solo con base en ese
   contexto (evita que el modelo invente datos).
3. **Interfaz** (`app.py`): interfaz web construida con Streamlit, con historial
   de conversación tipo chat y preguntas de ejemplo.
4. **Deploy**: la aplicación se ejecuta en una instancia de OCI Compute,
   accesible públicamente (ver sección de evidencia más abajo).

## Tecnologías utilizadas

| Componente | Herramienta |
|---|---|
| Lenguaje | Python 3.11 |
| Orquestación del agente | LangChain |
| Lectura de PDF | PyPDF (`PyPDFLoader`) |
| Embeddings | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`, local, sin costo) |
| Modelo de lenguaje | Groq (`llama-3.1-8b-instant`) |
| Índice vectorial | FAISS |
| Interfaz | Streamlit |
| Prototipado | Google Colab |
| Despliegue | OCI Compute |

## Instrucciones para ejecutar el proyecto

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone https://github.com/<tu-usuario>/agente-documental-clinica-vitalis.git
cd agente-documental-clinica-vitalis
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar la clave de API

```bash
cp .env.example .env
# Editar .env y colocar tu GROQ_API_KEY
```

### 3. Construir el índice vectorial (una sola vez)

```bash
python src/ingest.py
```

### 4. Ejecutar el agente

```bash
streamlit run app.py
```

Esto abre la interfaz web en el navegador, donde se pueden escribir
preguntas y el agente responde con base en el documento.

## Ejemplos de preguntas y respuestas

> Las respuestas siguientes se muestran a modo de ejemplo del
> comportamiento esperado del agente. *(Reemplazar con capturas o
> salidas reales una vez ejecutado el agente.)*

**Pregunta:** ¿Cuánto tiempo antes debo llegar a mi cita?
**Respuesta:** Se recomienda llegar 20 minutos antes de la hora
agendada. Si el retraso supera los 15 minutos sobre la hora de la
cita, esta puede reasignarse a otro paciente en espera.

**Pregunta:** ¿Qué pasa si cancelo mi cita con menos de 24 horas de anticipación?
**Respuesta:** Se genera un cargo administrativo equivalente al 20%
del costo de la consulta, salvo casos de urgencia médica justificada.

**Pregunta:** ¿Qué cobertura tiene el convenio con Seguros del Valle?
**Respuesta:** Cubre el 100% de la consulta general y el 80% de
especialidades, y requiere autorización previa para especialidades.

**Pregunta:** ¿Necesito ayuno antes de mi consulta?
**Respuesta:** Solo si la consulta incluye laboratorios de glucosa,
perfil lipídico o función hepática, en cuyo caso se requiere ayuno de
8 horas.

## Evidencia del deploy en OCI

*(Completar tras el despliegue.)*

- Enlace público de la aplicación: `<pendiente>`
- Captura de pantalla: `docs/screenshot-oci.png` *(agregar la imagen a esta ruta)*

## Estructura del repositorio

```
agente-documental-clinica-vitalis/
├── data/
│   └── clinica_vitalis_documentacion.pdf
├── src/
│   ├── ingest.py
│   └── agent.py
├── app.py
├── .streamlit/
│   └── config.toml
├── vector_store/          # generado automáticamente, no se sube a git
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
