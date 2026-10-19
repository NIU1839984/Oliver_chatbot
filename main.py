from fastapi import FastAPI
import uvicorn
import ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="Backend Oliver Chatbot - 01 INFORMATICA")

import sqlite3

# --- 1. ENDPOINT DE VERIFICACIÓN (CONECTADO A SQLITE) ---
@app.get("/verificar_cliente/{telefono}")
def verificar_cliente(telefono: str):
    # Conectamos con el archivo que acabamos de crear
    conexion = sqlite3.connect("clientes.db")
    cursor = conexion.cursor()
    
    # Buscamos el teléfono exacto en la tabla
    cursor.execute("SELECT nombre, privilegio FROM usuarios WHERE telefono = ?", (telefono,))
    resultado = cursor.fetchone()
    
    # Cerramos la conexión para no bloquear el archivo
    conexion.close()
    
    # Lógica de decisión
    if resultado:
        nombre_cliente, nivel_privilegio = resultado
        return {
            "estado": "registrado",
            "privilegio": nivel_privilegio,
            "mensaje": f"Hola {nombre_cliente}, hemos detectado un privilegio: {nivel_privilegio}."
        }
    else:
        return {
            "estado": "no_registrado",
            "privilegio": "denegado",
            "mensaje": "Acceso denegado. Para usar a Oliver, regístrate en 01 INFORMATICA."
        }

# --- 2. NUEVO ENDPOINT: RAG Y RESPUESTA DE LA IA ---
@app.get("/soporte_tecnico/")
def soporte_tecnico(pregunta: str):
    print("1. Conectando con la base de datos de manuales (ChromaDB)...")
    # Cargamos las herramientas matemáticas locales
    embeddings_locales = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_locales)
    
    # Buscamos los 3 fragmentos del manual que más se parezcan a la pregunta del usuario
    resultados = vector_db.similarity_search(pregunta, k=3)
    
    # Unimos esos fragmentos en un solo texto (Esto es el "Contexto")
    contexto_manual = "\n\n".join([doc.page_content for doc in resultados])
    
    print("2. Formulando el paquete para la IA (Ollama)...")
    # CAPA 1: Guardrails de Seguridad + Prompt Engineering
    system_prompt = f"""
    Eres Oliver, el asistente técnico exclusivo de la empresa 01 INFORMATICA.
    Tu objetivo es resolver dudas de reparación y soporte técnico basándote ÚNICAMENTE en la información de este manual.
    Si el usuario te pregunta cosas fuera de la informática (recetas, política, clima), debes negarte a responder y decir que solo ayudas con productos de 01 INFORMATICA.
    Si la respuesta a la duda no está en el manual adjunto, di: "No he encontrado la solución en el manual. Hemos derivado tu consulta al equipo técnico."
    
    AQUÍ TIENES LOS FRAGMENTOS DEL MANUAL DEL CLIENTE:
    {contexto_manual}
    """
    
    print("3. Generando la respuesta final...")
    # Llamamos a Llama 3.2 pasándole las instrucciones y la pregunta
    respuesta_ia = ollama.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': pregunta}
    ])
    
    # Devolvemos la respuesta formateada a la página web o WhatsApp
    return {
        "pregunta_del_cliente": pregunta,
        "respuesta_de_oliver": respuesta_ia['message']['content']
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)