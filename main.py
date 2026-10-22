from fastapi import FastAPI
import uvicorn
import ollama
import sqlite3
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="Backend Oliver Chatbot - 01 INFORMATICA")

from fastapi.middleware.cors import CORSMiddleware

# Permitir que el HTML se comunique con FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción se pondría el dominio real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. INICIALIZACIÓN GLOBAL ---
print("Cargando modelo matemático y bases de datos en memoria...")
embeddings_locales = OllamaEmbeddings(model="nomic-embed-text")

# SOLUCIÓN: Forzamos la matemática de "Coseno" para que no mida distancias falsas
cliente_faq = chromadb.PersistentClient(path="./faq_db")
faq_db = Chroma(
    client=cliente_faq, 
    collection_name="cache_oliver", 
    embedding_function=embeddings_locales,
    collection_metadata={"hnsw:space": "cosine"} # <--- LA MAGIA ESTÁ AQUÍ
)

cliente_rag = chromadb.PersistentClient(path="./chroma_db")
vector_db = Chroma(
    client=cliente_rag, 
    collection_name="langchain", 
    embedding_function=embeddings_locales
)
print("Bases de datos cargadas con éxito.")

# --- 1. VERIFICACIÓN DE CLIENTES (SQLITE) ---
@app.get("/verificar_cliente/{telefono}")
def verificar_cliente(telefono: str):
    conexion = sqlite3.connect("clientes.db")
    cursor = conexion.cursor()
    cursor.execute("SELECT nombre, privilegio FROM usuarios WHERE telefono = ?", (telefono,))
    resultado = cursor.fetchone()
    conexion.close()
    
    if resultado:
        return {"estado": "registrado", "privilegio": resultado[1], "mensaje": f"Hola {resultado[0]}."}
    return {"estado": "no_registrado", "privilegio": "denegado"}


# --- 2. SOPORTE TÉCNICO: RAG + AUTO-APRENDIZAJE ---
@app.get("/soporte_tecnico/")
def soporte_tecnico(pregunta: str):
    
    # FASE 1: BÚSQUEDA EN CACHÉ
    try:
        # Solo buscamos si el caché ya tiene algo guardado para evitar errores
        if faq_db._collection.count() > 0:
            resultado_cache = faq_db.similarity_search_with_score(pregunta, k=1)
            if resultado_cache:
                documento, score = resultado_cache[0]
                print(f"-> Análisis matemático (Caché): {score}")
                
                # Si la puntuación es baja (muy similar), disparamos la respuesta rápida
                if score < 0.4:
                    return {
                        "origen": "Caché Instantáneo (Aprendido)", 
                        "respuesta": documento.metadata["respuesta"]
                    }
    except Exception as e:
        print(f"Saltando caché por seguridad: {e}")
    
    
    # FASE 2: RAG CON LLAMA 3.2
    print("-> Pregunta no cacheada o nueva. Consultando el manual PDF...")
    resultados_pdf = vector_db.similarity_search(pregunta, k=3)
    contexto_manual = "\n\n".join([doc.page_content for doc in resultados_pdf])
    
    system_prompt = f"""
    Eres Oliver, el asistente técnico exclusivo de la empresa 01 INFORMATICA.
    Tu objetivo es resolver dudas de reparación basándote ÚNICAMENTE en el siguiente manual.
    Si la respuesta a la pregunta no aparece en este manual, debes decir EXACTAMENTE: "Lo siento, no he encontrado la solución en el manual."
    
    MANUAL:
    {contexto_manual}
    """
    
    respuesta_ia = ollama.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': pregunta}
    ])
    
    texto_final = respuesta_ia['message']['content']
    
    # FASE 3: APRENDIZAJE INTELIGENTE
    # Solo guardamos la respuesta en el caché si la IA supo encontrar la solución
    if "Lo siento, no he encontrado" not in texto_final:
        faq_db.add_texts(
            texts=[pregunta],
            metadatas=[{"respuesta": texto_final}]
        )
        print("-> Nueva respuesta aprendida y guardada en caché.")
    
    return {
        "origen": "IA Generativa (Llama 3.2)",
        "respuesta": texto_final
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)