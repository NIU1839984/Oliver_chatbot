from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
import ollama
import sqlite3
import json
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="Backend Oliver Chatbot - 01 INFORMATICA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("--- INICIALIZANDO SISTEMA OLIVER ESTRICTO ---")
embeddings_locales = OllamaEmbeddings(model="nomic-embed-text")

# 1. Base de datos del Caché Semántico
cliente_faq = chromadb.PersistentClient(path="./faq_db")
faq_db = Chroma(
    client=cliente_faq, 
    collection_name="cache_oliver", 
    embedding_function=embeddings_locales,
    collection_metadata={"hnsw:space": "cosine"}
)

# 2. Base de datos del RAG (Manuales)
cliente_rag = chromadb.PersistentClient(path="./chroma_db")
vector_db = Chroma(
    client=cliente_rag, 
    collection_name="langchain", 
    embedding_function=embeddings_locales
)
print("Bases de datos cargadas con éxito.")


@app.get("/verificar_cliente/{telefono}")
def verificar_cliente(telefono: str):
    conexion = sqlite3.connect("clientes.db")
    cursor = conexion.cursor()
    cursor.execute("SELECT nombre, privilegio, productos FROM usuarios WHERE telefono = ?", (telefono,))
    resultado = cursor.fetchone()
    conexion.close()
    
    if resultado:
        nombre, privilegio, productos_raw = resultado
        productos = json.loads(productos_raw) if productos_raw else []
        return {"estado": "registrado", "nombre": nombre, "privilegio": privilegio, "productos": productos, "mensaje": f"¡Hola {nombre}!"}
    return {"estado": "no_registrado", "mensaje": "Teléfono no encontrado en el sistema."}


@app.post("/derivar_incidencia/")
def derivar_incidencia(telefono: str = Form(...), consulta: str = Form(...)):
    conexion = sqlite3.connect("clientes.db")
    cursor = conexion.cursor()
    cursor.execute("INSERT INTO incidencias (telefono, consulta) VALUES (?, ?)", (telefono, consulta))
    conexion.commit()
    conexion.close()
    return {"estado": "ok", "mensaje": "Hemos derivado tu consulta al equipo técnico de 01 Informàtica."}


@app.get("/soporte_tecnico/")
def soporte_tecnico(pregunta: str):
    print(f"\n[NUEVA PREGUNTA]: {pregunta}")
    
    # --- FASE 1: CACHÉ (SIN ERRORES) ---
    try:
        if faq_db._collection.count() > 0:
            resultado_cache = faq_db.similarity_search_with_score(pregunta, k=1)
            if resultado_cache:
                documento, score = resultado_cache[0]
                respuesta_cacheada = documento.metadata["respuesta"]
                
                # Solo usa el caché si la nota es buena Y NO es un error guardado accidentalmente
                if score < 0.35 and "NO_SOLUCIONADO" not in respuesta_cacheada.upper():
                    print("-> ✅ Respondiendo desde el Caché Instantáneo.")
                    return {"origen": "Caché Instantáneo", "respuesta": respuesta_cacheada}
    except Exception:
        pass
    
    # --- FASE 2: RAG (Búsqueda en Manuales) ---
    resultados_pdf = vector_db.similarity_search(pregunta, k=3)
    contexto_manual = "\n\n".join([doc.page_content for doc in resultados_pdf])
    
    # --- FASE 3: LLAMA 3.2 (PROMPT DEFINITIVO EQUILIBRADO) ---
    system_prompt = f"""
    Eres Oliver, el experto de soporte técnico de 01 INFORMATICA. Tu objetivo es resolver los problemas tecnológicos de los clientes.
    
    CONTEXTO TÉCNICO (Manuales oficiales):
    {contexto_manual}
    
    REGLAS DE RESPUESTA:
    1. FLEXIBILIDAD TÉCNICA: Responde a CUALQUIER pregunta sobre informática, tecnología, hardware, software o redes. Usa el contexto técnico como base principal, pero SIÉNTETE LIBRE de usar tus conocimientos generales de informática para dar una solución completa y útil aunque la respuesta exacta no esté en los manuales.
    2. BARRERA DE DOMINIO: Si la pregunta NO tiene NADA que ver con la tecnología (ejemplo: recetas, croquetas, deportes, historia, chistes), está FUERA DE TU DOMINIO.
    3. REGLA DE RECHAZO: Si la pregunta está fuera de dominio (regla 2), TIENES ESTRICTAMENTE PROHIBIDO conversar, disculparte o dar explicaciones. Tu ÚNICA respuesta permitida es escribir exactamente esta palabra:
    NO_SOLUCIONADO
    """
    
    print("-> 🧠 Pensando respuesta (Modo Equilibrado)...")
    respuesta_ia = ollama.chat(model='llama3.2', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': pregunta}
    ])
    
    texto_final = respuesta_ia['message']['content'].strip()
    
    # --- CORTAFUEGOS DE LIMPIEZA DE PYTHON ---
    # Si la IA intenta ser amable al rechazar algo, forzamos la palabra clave pura.
    texto_upper = texto_final.upper()
    if "NO_SOLUCIONADO" in texto_upper or "NO SOLUCIONADO" in texto_upper:
        texto_final = "NO_SOLUCIONADO"
        
    # --- FASE 4: GUARDADO SEGURO (FORO Y CACHÉ) ---
    if texto_final != "NO_SOLUCIONADO":
        print("-> ✅ Respuesta técnica válida. Guardando en Caché y Foro.")
        faq_db.add_texts(texts=[pregunta], metadatas=[{"respuesta": texto_final}])
    else:
        print("-> ❌ Pregunta bloqueada por no ser de informática. Se activa derivación web.")
    
    return {"origen": "IA Generativa (Llama 3.2)", "respuesta": texto_final}


@app.get("/foro_faqs/")
def obtener_foro_faqs():
    try:
        datos = faq_db.get()
        faqs_publicas = []
        if datos and "documents" in datos:
            for i in range(len(datos["documents"])):
                respuesta = datos["metadatas"][i]["respuesta"]
                # Filtro final: no enviamos al foro nada que sea un error colado
                if "NO_SOLUCIONADO" not in respuesta.upper():
                    faqs_publicas.append({
                        "pregunta": datos["documents"][i],
                        "respuesta": respuesta
                    })
        return {"estado": "ok", "total": len(faqs_publicas), "faqs": faqs_publicas}
    except Exception as e:
        return {"estado": "error", "mensaje": str(e), "faqs": []}


@app.post("/whatsapp")
async def whatsapp_webhook(Body: str = Form(...), From: str = Form(...)):
    telefono_limpio = From.replace("whatsapp:", "").replace("+", "").strip()
    resultado = soporte_tecnico(pregunta=Body)
    respuesta_texto = resultado["respuesta"]
    
    if respuesta_texto == "NO_SOLUCIONADO":
        respuesta_texto = "Lo siento, no he encontrado la solución en mis manuales oficiales. He derivado tu consulta al equipo técnico de 01 INFORMATICA y te contactaremos pronto."
        derivar_incidencia(telefono=telefono_limpio, consulta=Body)
    
    twiml_response = f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Response><Message>{respuesta_texto}</Message></Response>"
    return Response(content=twiml_response, media_type="application/xml")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)