"""
medir_memoria.py
Mide cuánta memoria RAM (RSS) consume el proceso de Python antes y
después de cargar el agente completo (embeddings + FAISS + LLM chain),
para saber si cabe dentro de un límite dado (ej. 512 MB en Render,
1 GB en Streamlit Community Cloud) ANTES de desplegar.

Uso:
    python medir_memoria.py
"""
import os
import sys

try:
    import psutil
except ImportError:
    print("Falta psutil. Instálalo con: pip install psutil")
    sys.exit(1)

proceso = psutil.Process(os.getpid())


def memoria_actual_mb() -> float:
    """Devuelve la memoria RSS (memoria física real usada) del proceso en MB."""
    return proceso.memory_info().rss / (1024 * 1024)


def reportar(etiqueta: str, base_mb: float):
    actual = memoria_actual_mb()
    print(f"[{etiqueta}] RSS total: {actual:8.1f} MB   (delta desde el inicio: +{actual - base_mb:7.1f} MB)")
    return actual


def main():
    base = memoria_actual_mb()
    print(f"Memoria al arrancar el script (solo Python + imports base): {base:.1f} MB\n")

    # Importamos aquí (no arriba del archivo) para poder medir el impacto
    # de cada import por separado.
    reportar("Antes de cualquier import pesado", base)

    from dotenv import load_dotenv
    load_dotenv()
    reportar("Después de import dotenv + load_dotenv", base)

    import streamlit  # noqa: F401  (solo para medir su huella, igual que en producción)
    reportar("Después de import streamlit", base)

    from langchain_community.vectorstores import FAISS
    reportar("Después de import FAISS (langchain_community)", base)

    from langchain_community.embeddings import FastEmbedEmbeddings
    reportar("Después de import FastEmbedEmbeddings", base)

    from langchain_groq import ChatGroq  # noqa: F401
    reportar("Después de import langchain_groq", base)

    # Ahora sí, cargar el agente completo tal como lo hace agent.py
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from agent import cargar_agente  # type: ignore

    print("\nCargando embeddings + índice FAISS + cadena LLM (cargar_agente())...\n")
    qa_chain = cargar_agente()
    final_mb = reportar("DESPUÉS de cargar_agente() completo", base)

    print("\n" + "=" * 60)
    print(f"CONSUMO TOTAL DE RAM DEL PROCESO: {final_mb:.1f} MB")
    print("=" * 60)
    print("\nReferencia de límites de planes gratuitos:")
    print(f"  Render Free ......... 512 MB   -> {'CABE ✅' if final_mb < 512 else 'NO CABE ❌'}")
    print(f"  Streamlit Cloud ..... 1024 MB  -> {'CABE ✅' if final_mb < 1024 else 'NO CABE ❌'}")
    print("\n(Nota: esto mide solo el proceso base cargando el agente, sin el")
    print(" overhead adicional de Streamlit sirviendo requests concurrentes,")
    print(" así que en producción el uso real puede ser algo mayor. Deja un")
    print(" margen de seguridad de al menos 100-150 MB extra.)")


if __name__ == "__main__":
    main()
