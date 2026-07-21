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
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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
    # Inicializar embeddings y cargar base de datos vectorial
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        index_path, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Configurar el modelo
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Configurar el prompt
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # Crear cadena de combinación de documentos
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # Crear cadena de recuperación (estándar actual en LangChain 0.3.x)
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return qa_chain


def preguntar(qa_chain, pregunta: str) -> str:
    # Invocar la cadena utilizando la clave 'input'
    resultado = qa_chain.invoke({"input": pregunta})
    return resultado["answer"]


def main():
    print("Cargando agente de Clínica Vitalis... (Ctrl+C para salir)\n")
    try:
        qa_chain = cargar_agente()
    except Exception as e:
        print(f"Error al cargar el agente: {e}")
        return

    while True:
        try:
            pregunta = input("Pregunta: ").strip()
            if not pregunta:
                continue
            
            respuesta = preguntar(qa_chain, pregunta)
            print(f"Respuesta: {respuesta}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nHasta luego.")
            break
        except Exception as e:
            print(f"Ocurrió un error: {e}\n")


if __name__ == "__main__":
    main()