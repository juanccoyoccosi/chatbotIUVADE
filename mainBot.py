import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import json
import random
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from llama_cpp import Llama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import logging

# CONFIGURAR EL TOKEN DE TU BOT TELEGRAM
TELEGRAM_TOKEN = "7039997048:AAE-CKX6iw-636uz_Lqx8ZyLDPjdWBVHXYw"
base_url = "http://localhost:21465"

# Configuración del logging para depuración
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar el stemmer de NLTK
stemmer = LancasterStemmer()
nltk.download('punkt')

# Cargar datos de entrenamiento desde contenido.json
with open("contenido.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)

# Configuración del modelo Llama
model_path = r"C:\\Users\\TUF\\Desktop\\llama ia chatbot\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,  # Contexto reducido para mejor rendimiento
        n_threads=8,  # Ajusta según tu CPU
        verbose=False
    )
except Exception as e:
    logger.error(f"Error al cargar Llama: {e}")
    llm = None

def calcular_tokens_respuesta(prompt):
    """Calcula tokens necesarios según complejidad"""
    palabras = len(prompt.split())
    
    # Respuestas cortas para saludos y preguntas simples
    if palabras <= 3:
        return 30
    # Respuestas medias para preguntas básicas
    elif palabras <= 10:
        return 100
    # Respuestas largas para preguntas complejas
    else:
        return 300

def mejorar_clasificacion_entrada(entrada):
    """Mejora la clasificación con preprocesamiento"""
    entrada = entrada.lower().strip()
    entrada = ''.join(c for c in entrada if c.isalnum() or c.isspace())
    
    tokens = nltk.word_tokenize(entrada)
    
    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('spanish'))
    tokens = [t for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)

def calcular_umbral_dinamico(similitudes, n_mejores=3):
    """Calcula umbral basado en similitudes"""
    mejores_similitudes = np.sort(similitudes[0])[-n_mejores:]
    if len(mejores_similitudes) < 2:
        return 0.3
    
    if mejores_similitudes[-1] > 1.5 * mejores_similitudes[-2]:
        return mejores_similitudes[-1] * 0.8
    
    return 0.3


# Preparar datos para el modelo local
def preparar_datos():
    palabras, tags, auxX, auxY = [], [], [], []
    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
            # Tokenizar y stemear cada patrón
            tokens = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(patrones) if w.isalnum()]
            palabras.extend(tokens)
            auxX.append(tokens)
            auxY.append(contenido["tag"])
            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

    palabras = sorted(set(palabras))
    tags = sorted(tags)
    return palabras, tags, auxX, auxY


def crear_datos_entrenamiento(palabras, tags, auxX, auxY):
    entrenamiento, salida = [], []
    salidaVacia = [0] * len(tags)

    for x, doc in enumerate(auxX):
        cubeta = [1 if stemmer.stem(w.lower()) in doc else 0 for w in palabras]
        filaSalida = salidaVacia[:]
        filaSalida[tags.index(auxY[x])] = 1
        entrenamiento.append(cubeta)
        salida.append(filaSalida)

    return np.array(entrenamiento), np.array(salida)

def crear_modelo(shape_entrada, shape_salida):
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(shape_entrada,), activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(shape_salida, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

# Inicializar datos y modelo
palabras, tags, auxX, auxY = preparar_datos()
entrenamiento, salida = crear_datos_entrenamiento(palabras, tags, auxX, auxY)
modelo = crear_modelo(len(entrenamiento[0]), len(salida[0]))
modelo.fit(entrenamiento, salida, epochs=100, batch_size=10, verbose=1)

def obtener_respuesta_llama(prompt):
    """Genera respuesta adaptativa con Llama"""
    if llm is None:
        return None
        
    try:
        # Calcular longitud de respuesta necesaria
        max_tokens = calcular_tokens_respuesta(prompt)
        
        prompt_completo = f"""[INST] Actúa como IUVADITO, un asistente virtual amigable.
        Adapta el detalle de tu respuesta según la complejidad de la pregunta.
        Para preguntas simples, sé breve y directo.
        Para temas complejos, proporciona explicación detallada.
        
        Usuario: {prompt}
        
        IUVADITO:[/INST]"""
        
        respuesta = llm(
            prompt_completo,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            stop=["Usuario:", "\n", "[INST]"],
            repeat_penalty=1.2
        )
        
        texto_respuesta = respuesta['choices'][0]['text'].strip()
        return texto_respuesta.replace('[/INST]', '').replace('[INST]', '').strip()
    except Exception as e:
        logger.error(f"Error en Llama: {e}")
        return None


def generar_clasificadores_tfidf(datos):
    """Crea una matriz TF-IDF para mejorar la clasificación de intenciones"""
    tags = []
    corpus = []

    for contenido in datos["contenido"]:
        tags.append(contenido["tag"])
        patrones_concatenados = " ".join(contenido["patrones"])
        corpus.append(patrones_concatenados.lower())

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return vectorizer, tfidf_matrix, tags

# Inicializar matriz TF-IDF una vez para no recalcular siempre
vectorizer, tfidf_matrix, tags = generar_clasificadores_tfidf(datos)

def procesar_entrada_local(entrada):
    """Clasificación con longitud adaptativa"""
    entrada_procesada = mejorar_clasificacion_entrada(entrada)
    entrada_vectorizada = vectorizer.transform([entrada_procesada])
    similitudes = cosine_similarity(entrada_vectorizada, tfidf_matrix)
    
    umbral = calcular_umbral_dinamico(similitudes)
    indice_mejor = similitudes.argmax()
    confianza = similitudes[0, indice_mejor]
    
    palabras_entrada = len(entrada.split())
    # Ajustar umbral según complejidad
    if palabras_entrada <= 3:
        umbral *= 1.2  # Más estricto para preguntas simples
    
    if confianza > umbral:
        return tags[indice_mejor], confianza
    return None, 0
def obtener_respuesta_local(tag):
    """Busca la mejor respuesta en el archivo JSON"""
    for tagAux in datos["contenido"]:
        if tagAux["tag"] == tag:
            return random.choice(tagAux["respuestas"])
    return None

# Enviar un mensaje usando MyZap
def enviar_mensaje(session, phone, message):
    url = f"{base_url}/sendMessage"
    payload = {
        "session": session,
        "phone": phone,
        "message": message
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Lanza una excepción si la respuesta no es 200 OK
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error enviando mensaje: {e}")
        return None
# === MANEJADORES DE TELEGRAM ===
async def start(update: Update, context):
    # Envía un mensaje de bienvenida a WhatsApp usando MyZap
    phone_number = update.message.chat.id  # Asegúrate de obtener el número de teléfono del usuario
    session_id = "mysession"  # Cambia esto por tu ID de sesión de MyZap
    mensaje_bienvenida = "¡Hola! Soy IUVADITO 🤖. Pregúntame lo que quieras."
    
    # Enviar el mensaje de bienvenida
    enviar_mensaje(session_id, phone_number, mensaje_bienvenida)
    await update.message.reply_text(mensaje_bienvenida)


async def responder(update: Update, context: CallbackContext):
    entrada = update.message.text
    user_data = context.user_data
    phone_number = update.message.chat.id  # Asegúrate de obtener el número de teléfono del usuario
    session_id = "mysession"  # Cambia esto por tu ID de sesión de MyZap

    if "en_menu" in user_data and user_data["en_menu"]:
        if entrada.isdigit():
            opcion = entrada
            if opcion in user_data["opciones_menu"]:
                tag = user_data["opciones_menu"][opcion]
                respuesta = obtener_respuesta_local(tag)
                await update.message.reply_text(respuesta)
                # Envía la respuesta a WhatsApp usando MyZap
                enviar_mensaje(session_id, phone_number, respuesta)
                user_data["en_menu"] = False
                return
            await update.message.reply_text("Opción no válida. Por favor, elige un número del menú.")
            return
        await update.message.reply_text("Por favor, elige una opción numérica.")
        return

    tag, confianza = procesar_entrada_local(entrada)
    logger.info(f"Entrada: {entrada} | Tag: {tag} | Confianza: {confianza}")
    
    if tag and confianza > 0.3:
        respuesta = obtener_respuesta_local(tag)
        # Actualizar JSON si la confianza es alta
        if confianza > 0.7:
            actualizar_json(entrada, tag)
    else:
        respuesta = obtener_respuesta_llama(entrada) if llm else None
        if not respuesta:
            respuesta = "Disculpa, no entendí. ¿Podrías reformular tu mensaje?"

    for contenido in datos["contenido"]:
        if contenido["tag"] == tag and contenido.get("es_menu", False):
            user_data["en_menu"] = True
            user_data["opciones_menu"] = contenido["opciones_menu"]
            break

    await update.message.reply_text(respuesta)
    # Envía la respuesta a WhatsApp usando MyZap
    enviar_mensaje(session_id, phone_number, respuesta)
def actualizar_json(entrada, tag):
    try:
        with open("contenido.json", 'r+', encoding='utf-8') as archivo:
            datos_actuales = json.load(archivo)
            for contenido in datos_actuales["contenido"]:
                if contenido["tag"] == tag and entrada not in contenido["patrones"]:
                    contenido["patrones"].append(entrada)
                    archivo.seek(0)
                    json.dump(datos_actuales, archivo, indent=4, ensure_ascii=False)
                    archivo.truncate()
                    
                    # Actualizar clasificadores
                    global vectorizer, tfidf_matrix, tags
                    vectorizer, tfidf_matrix, tags = generar_clasificadores_tfidf(datos_actuales)
                    return
    except Exception as e:
        logger.error(f"Error actualizando JSON: {e}")
    
# === INICIAR BOT DE TELEGRAM ===
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Agregar manejadores correctamente
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))

    print("¡IUVADITO está activo en Telegram!")
    app.run_polling()

if __name__ == "__main__":
    main()
    