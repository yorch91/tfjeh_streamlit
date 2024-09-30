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

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key

# Lista de ejercicios aleatorios
ejercicios_aleatorios = [
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

    # Input de ejercicio proporcionado por el usuario
    ejercicio_usuario = st.text_area("Ingresa un ejercicio de programación en C para recibir sugerencias:")

    # Botón para generar ejercicio aleatorio
    if st.button("Generar ejercicio aleatorio"):
        ejercicio_aleatorio = random.choice(ejercicios_aleatorios)
        st.write(f"Ejercicio generado: {ejercicio_aleatorio}")
        ejercicio_usuario = ejercicio_aleatorio  # Al seleccionar un ejercicio aleatorio, se usa como input

    # Si el usuario ingresa un ejercicio o se genera uno, el chatbot analiza el ejercicio
    if ejercicio_usuario:
        # Plantilla de prompt para interpretar el ejercicio
        prompt_template = """
        Eres un profesor experto en el lenguaje de programación C. 
        El siguiente es un ejercicio de programación que un estudiante quiere resolver:
        {ejercicio_usuario}

        Tu tarea es ayudar al estudiante a razonar sobre el ejercicio sin proporcionar directamente el código de la solución.
        Ofrece sugerencias sobre cómo abordarlo, posibles estructuras de datos a usar, y otros puntos importantes.
        Puedes incluir ejemplos de código que ayuden al razonamiento, pero no debes resolver el ejercicio por completo.
        Ve desglozando el ejercicio en las partes mas importantes y a continuacion sugiere codigo y conceptos a revisar
        Tu respuesta debe sugerir conceptos y estructuras de codigo a utilizar, pero no deben resolver el ejercicio, sino solo servir de guia
        
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
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
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

# Llamar la función para mostrar la página
mostrar_pagina_ejercicios()
