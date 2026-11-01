from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import os

print("--- INICIANDO INYECCIÓN DE CONOCIMIENTOS ---")

# 1. CARGAR EL PDF
nombre_del_pdf = "manuales/TL-WN722N(EU)_V2_UG_1476755166109v.pdf"  # <--- PON AQUÍ EL NOMBRE EXACTO DE TU PDF

if not os.path.exists(nombre_del_pdf):
    print(f"❌ ERROR: No encuentro el archivo '{nombre_del_pdf}'. Ponlo en la carpeta Manuales.")
    exit()

print(f"📖 Leyendo el archivo {nombre_del_pdf}...")
loader = PyPDFLoader(nombre_del_pdf)
docs = loader.load()

# 2. TROCEAR EL TEXTO
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(f"✂️ Texto troceado en {len(splits)} fragmentos matemáticos.")

# 3. GUARDAR EN LA BASE DE DATOS (CHROMA DB)
print("🧠 Conectando con el motor de IA local (Ollama)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
cliente_rag = chromadb.PersistentClient(path="./chroma_db")

vector_db = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    client=cliente_rag,
    collection_name="langchain"
)

print("✅ ¡ÉXITO! Conocimiento inyectado para siempre en la carpeta 'chroma_db'.")