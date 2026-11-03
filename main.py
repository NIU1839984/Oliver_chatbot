from fastapi import FastAPI, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
import ollama
import sqlite3
import json
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from typing import Optional

app = FastAPI(title="Backend Oliver Chatbot - 01 INFORMATICA")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("--- INICIALIZANDO SISTEMA OLIVER DEFINITIVO ---")
embeddings_locales = OllamaEmbeddings(model="nomic-embed-text")

# 1. Base de datos del Caché Semántico (Supervisado)
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
def soporte_tecnico(pregunta: str, historial: Optional[str] = Query(None)):
    # BLINDAJE: Quitamos espacios invisibles al principio y al final
    pregunta = pregunta.strip()
    
    print(f"\n" + "="*50)
    print(f"[NUEVA PREGUNTA DEL CLIENTE]: {pregunta}")
    # ... resto del código sin tocar nada más
    
    # --- FASE 1: CACHÉ (SIN ERRORES) ---
    try:
        if faq_db._collection.count() > 0:
            resultado_cache = faq_db.similarity_search_with_score(pregunta, k=1)
            if resultado_cache:
                documento, score = resultado_cache[0]
                respuesta_cacheada = documento.metadata["respuesta"]
                print(f"Score de la pregunta: {score:.4f} | Respuesta cacheada: {respuesta_cacheada}")
                # Solo usa el caché si la nota es buena Y NO es un error guardado accidentalmente
                if score < 0.2 and "NO_SOLUCIONADO" not in respuesta_cacheada.upper():
                    print("-> ✅ Respondiendo desde el Caché Instantáneo.")
                    return {"origen": "Caché Instantáneo", "respuesta": respuesta_cacheada}
    except Exception:
        pass
    
    # --- FASE 2: RAG (Búsqueda en Manuales) ---
    resultados_pdf = vector_db.similarity_search(pregunta, k=3)
    contexto_manual = "\n\n".join([doc.page_content for doc in resultados_pdf])
    
    # --- FASE 3: PREPARAR MEMORIA Y PROMPT EQUILIBRADO ---
    mensajes_chat = []
    
    system_prompt = f"""
    Eres Oliver, el experto de soporte técnico de 01 INFORMATICA. Tu objetivo es resolver los problemas tecnológicos de los clientes teniendo en cuenta la conversación anterior.
    
    CONTEXTO TÉCNICO (Manuales oficiales):
    {contexto_manual}
    
    REGLAS DE RESPUESTA:
    1. FLEXIBILIDAD TÉCNICA: Responde a CUALQUIER pregunta sobre informática, tecnología, hardware, software o redes. Usa el contexto técnico como base principal, pero SIÉNTETE LIBRE de usar tus conocimientos generales de informática para dar una solución completa y útil.
    2. BARRERA DE DOMINIO: Si la pregunta NO tiene NADA que ver con la tecnología (ejemplo: recetas, croquetas, deportes, historia, chistes), está FUERA DE TU DOMINIO.
    3. REGLA DE RECHAZO: Si la pregunta está fuera de dominio (regla 2), TIENES ESTRICTAMENTE PROHIBIDO conversar, disculparte o dar explicaciones. Tu ÚNICA respuesta permitida es escribir exactamente esta palabra:
    NO_SOLUCIONADO
    """
    mensajes_chat.append({'role': 'system', 'content': system_prompt})
    
    # Inyectar el historial de conversación (memoria) si existe
    if historial:
        try:
            historial_lista = json.loads(historial)
            for msg in historial_lista:
                mensajes_chat.append({'role': msg["rol"], 'content': msg["contenido"]})
            print(f"-> 📜 Memoria inyectada: {len(historial_lista)} mensajes anteriores.")
        except json.JSONDecodeError:
            print("-> ⚠️ Error al procesar el historial.")
            
    mensajes_chat.append({'role': 'user', 'content': pregunta})
    
    print("-> 🧠 Pensando respuesta (Modo Equilibrado + Memoria)...")
    respuesta_ia = ollama.chat(model='llama3.2', messages=mensajes_chat)
    
    texto_final = respuesta_ia['message']['content'].strip()
    
    # --- CORTAFUEGOS DE LIMPIEZA DE PYTHON ---
    texto_upper = texto_final.upper()
    if "NO_SOLUCIONADO" in texto_upper or "NO SOLUCIONADO" in texto_upper:
        texto_final = "NO_SOLUCIONADO"
        print("-> ❌ Pregunta bloqueada por no ser de informática. Se activa derivación web.")
        
    # NOTA: Ya no hay Fase 4 de guardado aquí. Se delega al botón "Sí, gracias".
    return {"origen": "IA Generativa (Llama 3.2)", "respuesta": texto_final}


# --- NUEVO ENDPOINT: APRENDIZAJE SUPERVISADO ---
# --- ENDPOINT CON CURADOR Y REESCRITOR DE FAQS ---
# --- ENDPOINT CON CURADOR Y REESCRITOR DE FAQS (MEJORADO) ---
@app.post("/confirmar_solucion/")
def confirmar_solucion(pregunta: str = Form(...), respuesta: str = Form(...)):
    print(f"\n[CURADOR DE FAQS]: Auditando calidad y contexto de la solución...")
    
    # 1. Descarte rápido si es una respuesta de error
    if "NO_SOLUCIONADO" in respuesta.upper():
        print("-> 🗑️ Rechazado automáticamente: Contiene NO_SOLUCIONADO.")
        return {"estado": "desechado"}

    system_prompt = """
    Eres el Curador de Conocimiento de 01 INFORMATICA. Tu trabajo es optimizar pares de (Pregunta, Respuesta) para un Foro de FAQs.

    INSTRUCCIONES ESTRICTAS:
    1. Si la respuesta NO soluciona el problema o es un mensaje de error -> Devuelve "DESECHABLE".
    2. Si la respuesta ES útil, evalúa la pregunta del usuario:
       - Si es corta, ambigua o depende del historial (ej: "¿Y qué hago ahora?", "No va", "Se me apaga"), REESCRÍBELA.
       - REGLA DE ORO: La reescritura DEBE ser una pregunta real, formulada desde el punto de vista del usuario (ej: "¿Cómo soluciono...", "¿Por qué el router...", "¿Qué significa la luz...").
       - TIENES ESTRICTAMENTE PROHIBIDO poner pasos a seguir, instrucciones o resúmenes de la solución dentro de la pregunta. La pregunta solo expone el síntoma o la duda.
       - Obligatorio usar signos de interrogación (¿?).
       - Si la pregunta original ya es perfecta y autónoma, devuélvela exactamente igual.

    FORMATO OBLIGATORIO DE SALIDA (ÚNICAMENTE JSON VÁLIDO):
    {"estado": "VALIDO", "pregunta_final": "¿Tu pregunta bien formulada aquí?"}
    o
    {"estado": "DESECHABLE"}
    """
    
    user_prompt = f"PREGUNTA DEL USUARIO: {pregunta}\nRESPUESTA GENERADA: {respuesta}"
    
    try:
        evaluacion_ia = ollama.chat(
            model='llama3.2',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            format='json'
        )
        
        contenido = json.loads(evaluacion_ia['message']['content'])
        
        if contenido.get("estado") == "VALIDO" and "pregunta_final" in contenido:
            pregunta_optimizada = contenido["pregunta_final"]
            
            faq_db.add_texts(texts=[pregunta_optimizada], metadatas=[{"respuesta": respuesta}])
            print(f"-> 🏆 FAQ APROBADA Y OPTIMIZADA PARA EL FORO:\n   Pregunta Original: '{pregunta}'\n   Pregunta Foro: '{pregunta_optimizada}'")
            return {"estado": "guardado", "pregunta_optimizada": pregunta_optimizada}
        else:
            print("-> 🗑️ FAQ DESECHADA POR EL CURADOR: La respuesta no aporta suficiente valor.")
            return {"estado": "desechado"}
            
    except Exception as e:
        print(f"-> ⚠️ Error en el procesamiento del Curador: {e}")
        faq_db.add_texts(texts=[pregunta], metadatas=[{"respuesta": respuesta}])
        return {"estado": "guardado_fallback"}
        
@app.get("/foro_faqs/")
def obtener_foro_faqs():
    try:
        datos = faq_db.get()
        faqs_publicas = []
        if datos and "documents" in datos:
            for i in range(len(datos["documents"])):
                respuesta = datos["metadatas"][i]["respuesta"]
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
    else:
        # En WhatsApp no hay botón de confirmación, guardamos automáticamente
        faq_db.add_texts(texts=[Body], metadatas=[{"respuesta": respuesta_texto}])
        
    twiml_response = f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<Response><Message>{respuesta_texto}</Message></Response>"
    return Response(content=twiml_response, media_type="application/xml")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)