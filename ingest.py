import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

print("1. Buscando manuales PDF en la carpeta './manuales'...")
# Usamos pypdf para leer todos los PDFs de la carpeta especificada
loader = PyPDFDirectoryLoader("./manuales")
documentos = loader.load()

if not documentos:
    print("Error: No se ha encontrado ningún PDF en la carpeta 'manuales'.")
else:
    print(f"-> Se han cargado {len(documentos)} páginas en total.")

    print("\n2. Troceando los manuales en fragmentos pequeños (Tokens)...")
    # Dividimos el texto en trozos de 1000 caracteres para no saturar a la IA
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = text_splitter.split_documents(documentos)
    print(f"-> Los manuales se han dividido en {len(fragmentos)} fragmentos.")

    print("\n3. Convirtiendo texto a vectores y guardando en ChromaDB...")
    # Usamos Ollama para transformar el texto en coordenadas matemáticas
    # Nota: Usaremos el modelo llama3.2 que ya tenéis instalado
    embeddings_locales = OllamaEmbeddings(model="nomic-embed-text")
    
    # Creamos la base de datos y la guardamos en el disco duro local
    vector_db = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings_locales,
        persist_directory="./chroma_db"
    )
    
    print("\n¡ÉXITO! Base de datos vectorial de 01 INFORMATICA creada correctamente en './chroma_db'.")