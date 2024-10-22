import os
import streamlit as st
import random
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key

if api_key is None:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]  # Para producción en Streamlit Cloud
    except KeyError:
        raise ValueError("API Key de OpenAI no encontrada en el entorno local ni en Streamlit Secrets")

# Lista de ejercicios base para la generación aleatoria
ejercicios_base = [
    "Escribe un programa en C que invierta un array de enteros.",
    "Escribe un programa en C que implemente la búsqueda binaria en un array.",
    "Crea un programa en C que use punteros para intercambiar dos variables.",
    "Escribe un programa en C que calcule el factorial de un número utilizando recursión.",
    "Crea un programa en C que lea un archivo de texto y cuente el número de líneas."
]

# Asegúrate de tener una función que cargue los documentos en FAISS (db)
def cargar_documentos():
    loader = TextLoader("unidades_completas.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Función para mostrar la página de "Interpretar ejercicio"
def mostrar_pagina_ejercicios():
    # Cargar la base de datos de documentos
    db = cargar_documentos()

    # Encabezado de la página
    st.header("Interpretar ejercicio y sugerencias")

    # Input para que el usuario escriba una base de texto para generar un ejercicio aleatorio
    base_texto = st.text_input("Ingresa un texto base para generar un ejercicio aleatorio:")

    # Campo para mostrar el ejercicio generado
    ejercicio_generado = st.empty()

    # Botón para generar ejercicio aleatorio basado en el texto base ingresado
    if st.button("Generar ejercicio aleatorio"):
        if base_texto.strip():
            # Generar ejercicio aleatorio combinando con el texto base
            ejercicio_aleatorio = f"{base_texto} {random.choice(ejercicios_base)}"
            ejercicio_generado.text_area("Ejercicio generado:", value=ejercicio_aleatorio, height=100)
        else:
            st.warning("Por favor ingresa un texto base para generar un ejercicio.")

    # Input del ejercicio proporcionado por el usuario (máximo 100 caracteres)
    ejercicio_usuario = st.text_area("Ingresa un ejercicio de programación en C (máx. 100 caracteres):", max_chars=100)

    # Botón para analizar el ejercicio ingresado
    if st.button("Analizar ejercicio"):
        if ejercicio_usuario.strip():
            # Plantilla de prompt optimizada para reducir costos
            prompt_template = """
            Eres un profesor de C. El estudiante necesita ayuda con el siguiente ejercicio:
            {ejercicio_usuario}

            Ayúdalo a razonarlo, sugiriendo estructuras y conceptos clave, pero sin dar la solución completa.

            Documentos relacionados:
            {context}
            """

            # Crear el PromptTemplate
            prompt = PromptTemplate(
                input_variables=["context", "ejercicio_usuario"],
                template=prompt_template
            )

            try:
                # Buscar documentos relevantes en FAISS
                docs = db.similarity_search(ejercicio_usuario)

                # Combinar los documentos relevantes en un solo string como contexto
                context = "\n\n".join([doc.page_content for doc in docs])

                # Crear el LLM para analizar el ejercicio
                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
                chain = LLMChain(llm=llm, prompt=prompt)

                # Ejecutar la cadena pasando el contexto y el ejercicio del usuario
                answer = chain.run(context=context, ejercicio_usuario=ejercicio_usuario)

                # Si la respuesta está vacía, mostrar un mensaje
                if not answer.strip():
                    st.warning("No se pudo analizar el ejercicio. Por favor, intenta con otro ejercicio.")
                    return

                # Mostrar las sugerencias y el análisis del ejercicio
                st.write("Sugerencias y análisis del ejercicio:")
                st.write(answer)

            except Exception as e:
                st.error(f"Hubo un error al procesar tu ejercicio: {e}")
        else:
            st.warning("Por favor, ingresa un ejercicio antes de analizar.")

# Llamar la función para mostrar la página
mostrar_pagina_ejercicios()
