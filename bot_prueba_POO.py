import telebot
from telebot import types
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
import json
import random
from llama_cpp import Llama

# Configuración del bot de Telegram
TOKEN = '7039997048:AAE-CKX6iw-636uz_Lqx8ZyLDPjdWBVHXYw'
bot = telebot.TeleBot(TOKEN)

# Inicializar NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
stemmer = LancasterStemmer()

# Configurar Llama
model_path = r"C:\\Users\\TUF\\Desktop\\llama ia chatbot\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

try:
    llm = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_threads=8,
        verbose=False
    )
except Exception as e:
    print(f"Error al cargar el modelo Llama: {e}")
    llm = None

# Cargar datos de entrenamiento
with open("contenido.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)

# Preparar datos y modelo
def preparar_datos():
    palabras = []
    tags = []
    auxX = []
    auxY = []

    for contenido in datos["contenido"]:
        for patrones in contenido["patrones"]:
            auxPalabra = nltk.word_tokenize(patrones)
            palabras.extend(auxPalabra)
            auxX.append(auxPalabra)
            auxY.append(contenido["tag"])
            if contenido["tag"] not in tags:
                tags.append(contenido["tag"])

    palabras = [stemmer.stem(w.lower()) for w in palabras if w != "?"]
    palabras = sorted(list(set(palabras)))
    tags = sorted(tags)

    return palabras, tags, auxX, auxY

def crear_datos_entrenamiento(palabras, tags, auxX, auxY):
    entrenamiento = []
    salida = []
    salidaVacia = [0 for _ in range(len(tags))]

    for x, documento in enumerate(auxX):
        cubeta = []
        auxPalabra = [stemmer.stem(w.lower()) for w in documento]
        for w in palabras:
            if w in auxPalabra:
                cubeta.append(1)
            else:
                cubeta.append(0)
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

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo

# Inicializar modelo
palabras, tags, auxX, auxY = preparar_datos()
entrenamiento, salida = crear_datos_entrenamiento(palabras, tags, auxX, auxY)
modelo = crear_modelo(len(entrenamiento[0]), len(salida[0]))
modelo.fit(entrenamiento, salida, epochs=100, batch_size=10, verbose=1)

def obtener_respuesta_llama(prompt, nombre_usuario):
    """Obtiene una respuesta del modelo Llama"""
    if llm is None:
        return "Lo siento, el modelo no está disponible actualmente."

    try:
        prompt_completo = f"""[INST] Eres un asistente amigable llamado IUVADITO que responde en español al cliente {nombre_usuario}.
        
Usuario: {prompt}

IUVADITO: [/INST]"""

        respuesta = llm(
            prompt_completo,
            max_tokens=50,
            temperature=0.7,
            stop=["Usuario:", "\n\n"]
        )

        return respuesta['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error con Llama: {e}")
        return "Lo siento, estoy teniendo problemas para procesar tu solicitud."

def procesar_entrada_local(entrada):
    """Procesa la entrada del usuario usando el modelo local"""
    cubeta = [0 for _ in range(len(palabras))]
    entradaProcesada = nltk.word_tokenize(entrada)
    entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]

    for palabraIndividual in entradaProcesada:
        for i, palabra in enumerate(palabras):
            if palabra == palabraIndividual:
                cubeta[i] = 1

    resultados = modelo.predict(np.array([cubeta]), verbose=0)
    resultadosIndices = np.argmax(resultados[0])
    confianza = resultados[0][resultadosIndices]

    return tags[resultadosIndices], confianza

def obtener_respuesta_local(tag):
    """Obtiene una respuesta local basada en el tag"""
    for tagAux in datos["contenido"]:
        if tagAux["tag"] == tag:
            return random.choice(tagAux["respuestas"])
    return None

# Manejadores de Telegram
@bot.message_handler(commands=['start'])
def send_welcome(message):
    nombre_usuario = message.from_user.first_name
    bot.reply_to(message, f"¡Hola {nombre_usuario}! Soy IUVADITO. ¿En qué puedo ayudarte hoy?")

@bot.message_handler(func=lambda message: True)
def responder_mensaje(message):
    entrada = message.text
    nombre_usuario = message.from_user.first_name
    
    # Primero intentamos con el modelo local
    tag, confianza = procesar_entrada_local(entrada)
    
    if confianza > 0.7:
        respuesta = obtener_respuesta_local(tag)
        bot.reply_to(message, respuesta)
    else:
        # Si no hay suficiente confianza, usamos Llama
        respuesta_llama = obtener_respuesta_llama(entrada, nombre_usuario)
        bot.reply_to(message, respuesta_llama)

# Iniciar el bot
if __name__ == "__main__":
    print("Bot iniciado...")
    bot.polling(none_stop=True)