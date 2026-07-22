from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../data/clinica_vitalis_documentacion.pdf")
docs = loader.load()
texto_completo = "\n".join(d.page_content for d in docs)
idx = texto_completo.lower().find("embarazo")
print(texto_completo[max(0, idx-200):idx+300])