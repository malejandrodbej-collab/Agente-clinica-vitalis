import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vector_store")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
vs = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

resultados = vs.similarity_search("cuánto cuesta la prueba de embarazo", k=6)
for i, r in enumerate(resultados):
    print(f"--- Resultado {i+1} ---")
    print(r.page_content)
    print()