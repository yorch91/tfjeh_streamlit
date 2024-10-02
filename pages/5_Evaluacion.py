import os
import streamlit as st
import random
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Cargar las variables de entorno
if os.path.exists(".env"):
    load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key

if api_key is None:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]  # Para producción en Streamlit Cloud
    except KeyError:
        raise ValueError("API Key de OpenAI no encontrada en el entorno local ni en Streamlit Secrets")

# Cargar la base de datos FAISS una sola vez
def cargar_documentos():
    loader = TextLoader("unidades_completas.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Inicializar FAISS solo una vez
db = cargar_documentos()

# Función para generar nuevas preguntas
def generar_nuevas_preguntas(user_query, numero_de_preguntas):
    # Lista de temas aleatorios
    temas_aleatorios = ["Funciones", "Punteros", "Estructuras", "Operadores", "Manejo de memoria"]
    preguntas_generadas = []

    # Plantilla de prompt optimizada
    prompt_template = """
    Eres un experto en C. Genera {numero_de_preguntas} preguntas de opción múltiple combinando el tema {user_query} con {tema_aleatorio}.
    Si es posible, incluye preguntas de "¿Cuál es la salida de este código?" con un bloque de código C:
    ```
    #include <stdio.h>
    int main() {{
        // Código aquí
        return 0;
    }}
    ```
    Responde en formato JSON:
    [
        {{
            "pregunta": "La pregunta",
            "opciones": {{"A": "Opción A", "B": "Opción B", "C": "Opción C", "D": "Opción D"}},
            "respuesta_correcta": "Letra de la opción correcta"
        }},
        ...
    ]
    Documentos relacionados:
    {context}
    """

    # Crear el PromptTemplate
    prompt = PromptTemplate(
        input_variables=["context", "user_query", "tema_aleatorio", "numero_de_preguntas"],
        template=prompt_template
    )

    try:
        # Buscar documentos relevantes
        docs = db.similarity_search(user_query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Crear el LLM con temperature baja
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        chain = LLMChain(llm=llm, prompt=prompt)

        # Ejecutar la cadena pasando el contexto y los temas
        answer = chain.run(context=context, user_query=user_query, tema_aleatorio=random.choice(temas_aleatorios), numero_de_preguntas=numero_de_preguntas)

        # Parsear la respuesta como JSON
        try:
            preguntas = json.loads(answer)
            if not isinstance(preguntas, list):
                raise ValueError("Respuesta no es una lista.")
        except json.JSONDecodeError as e:
            st.error(f"Error al procesar la respuesta: {e}")
            return None

        # Asegurarse de que las preguntas sean distintas
        for pregunta in preguntas:
            if pregunta not in preguntas_generadas:
                preguntas_generadas.append(pregunta)

        return preguntas_generadas

    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None

# Función para mostrar el resultado final
def mostrar_resultado():
    correctas = sum(1 for idx, resp in enumerate(st.session_state["respuestas"]) if resp == st.session_state["preguntas"][idx]["respuesta_correcta"])
    incorrectas = len(st.session_state["respuestas"]) - correctas

    st.subheader("Resultados:")
    st.write(f"Respuestas correctas: {correctas}")
    st.write(f"Respuestas incorrectas: {incorrectas}")

# Función para reiniciar la evaluación
def reiniciar_evaluacion():
    st.session_state.clear()
    st.rerun()

# Función para mostrar la página de evaluación
def mostrar():
    st.header("Autoevaluacion")

    if "preguntas" not in st.session_state:
        st.session_state["preguntas"] = []
    if "respuestas" not in st.session_state:
        st.session_state["respuestas"] = []
    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""
    if "numero_de_preguntas" not in st.session_state:
        st.session_state["numero_de_preguntas"] = 3  # Limitar a 3 preguntas
    if "validado" not in st.session_state:
        st.session_state["validado"] = False

    # Input del usuario
    if not st.session_state["preguntas"]:
        user_query = st.text_input("Ingresa un tema para recibir preguntas:", max_chars=50)
        st.session_state["user_query"] = user_query

        numero_de_preguntas = st.number_input("¿Cuántas preguntas deseas generar?", min_value=1, max_value=3, value=3)
        st.session_state["numero_de_preguntas"] = numero_de_preguntas

        if st.button("Generar preguntas"):
            preguntas_nuevas = generar_nuevas_preguntas(st.session_state["user_query"], st.session_state["numero_de_preguntas"])
            if preguntas_nuevas:
                st.session_state["preguntas"] = preguntas_nuevas
                st.session_state["respuestas"] = [None] * len(preguntas_nuevas)

    # Mostrar preguntas
    if st.session_state["preguntas"]:
        for pregunta_idx, pregunta_actual in enumerate(st.session_state["preguntas"]):
            st.write(f"Pregunta {pregunta_idx + 1}: {pregunta_actual['pregunta']}")

            # Mostrar código si existe
            if "codigo" in pregunta_actual and pregunta_actual["codigo"].strip():
                st.code(pregunta_actual["codigo"], language='c')

            # Opciones
            opciones = [f"{key}) {value}" for key, value in pregunta_actual['opciones'].items()]
            selected_option = st.radio(f"Selecciona tu respuesta para la pregunta {pregunta_idx + 1}:", opciones, key=f"pregunta_{pregunta_idx}")

            if selected_option:
                st.session_state["respuestas"][pregunta_idx] = selected_option[0]

        # Validar respuestas
        if None not in st.session_state["respuestas"]:
            if st.button("Validar respuestas"):
                st.session_state["validado"] = True

        # Mostrar resultados
        if st.session_state["validado"]:
            mostrar_resultado()

            # Botón para reiniciar
            if st.button("Reiniciar evaluación"):
                reiniciar_evaluacion()

# Ejecutar la función de mostrar
mostrar()
