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

if os.path.exists(".env"):
    load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key

if api_key is None:
    try:
        api_key = st.secrets["OPENAI_API_KEY"]  # Para producción en Streamlit Cloud
    except KeyError:
        raise ValueError("API Key de OpenAI no encontrada en el entorno local ni en Streamlit Secrets")

# Función que carga los documentos en FAISS
def cargar_documentos():
    loader = TextLoader("unidades_completas.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Función para generar nuevas preguntas
def generar_nuevas_preguntas(user_query, numero_de_preguntas):
    db = cargar_documentos()
    temas_aleatorios = ["Funciones", "Punteros", "Estructuras", "Control de periféricos", "Operadores", "Manejo de memoria"]
    preguntas_generadas = []

    # Nuevo prompt para preguntas tipo "¿Cuál es la salida de este código?"
    prompt_template = """
    Eres un profesor experto en el lenguaje de programación C.
    Evita hacer referencias a C++, cin, cout, namespaces, o características específicas de C++.
    Genera {numero_de_preguntas} preguntas de opción múltiple del tipo "¿Cuál es la salida de este código?".
    Debes proporcionar el código en C y 4 opciones de respuesta, con una respuesta correcta. El formato JSON debe ser el siguiente:
    
    [
        {{
            "pregunta": "¿Cuál es la salida del siguiente código?",
            "codigo": "El código en C debe ir aquí.",
            "opciones": {{
                "A": "Opción A",
                "B": "Opción B",
                "C": "Opción C",
                "D": "Opción D"
            }},
            "respuesta_correcta": "La letra de la opción correcta y una breve explicación."
        }},
        ...
    ]
    
    Aquí tienes documentos de contexto:
    {context}
    """

    prompt = PromptTemplate(
        input_variables=["context", "user_query", "numero_de_preguntas"],
        template=prompt_template
    )

    try:
        docs = db.similarity_search(user_query)
        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        chain = LLMChain(llm=llm, prompt=prompt)
        answer = chain.run(context=context, user_query=user_query, numero_de_preguntas=numero_de_preguntas)

        try:
            preguntas = json.loads(answer)
            if not isinstance(preguntas, list):
                raise ValueError("La respuesta no es una lista.")
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Error al procesar las preguntas: {e}")
            return None

        for pregunta in preguntas:
            if pregunta not in preguntas_generadas:
                preguntas_generadas.append(pregunta)

        return preguntas_generadas

    except Exception as e:
        st.error(f"Error inesperado: {e}")
        return None

# Función para mostrar los resultados finales
def mostrar_resultado():
    correctas = sum(1 for idx, resp in enumerate(st.session_state["respuestas"]) if resp == st.session_state["preguntas"][idx]["respuesta_correcta"])
    incorrectas = len(st.session_state["respuestas"]) - correctas

    st.subheader("Resultados finales:")
    for idx, pregunta in enumerate(st.session_state["preguntas"]):
        st.write(f"Pregunta {idx + 1}: {pregunta['pregunta']}")
        st.write(f"Código:\n{pregunta['codigo']}")
        st.write(f"Tu respuesta: {st.session_state['respuestas'][idx]}")
        st.write(f"Respuesta correcta: {pregunta['respuesta_correcta']}")
        if st.session_state['respuestas'][idx] == pregunta['respuesta_correcta']:
            st.write("Resultado: Correcto ✅")
        else:
            st.write("Resultado: Incorrecto ❌")

    st.write(f"Respuestas correctas: {correctas}")
    st.write(f"Respuestas incorrectas: {incorrectas}")

# Función para reiniciar la evaluación
def reiniciar_evaluacion():
    st.session_state.clear()
    st.rerun()

# Función para mostrar la página de evaluación
def mostrar():
    st.header("Autoevaluación")

    if "preguntas" not in st.session_state:
        st.session_state["preguntas"] = []
    if "respuestas" not in st.session_state:
        st.session_state["respuestas"] = []
    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""
    if "numero_de_preguntas" not in st.session_state:
        st.session_state["numero_de_preguntas"] = 5
    if "validado" not in st.session_state:
        st.session_state["validado"] = False

    if not st.session_state["preguntas"]:
        user_query = st.text_input("Ingresa un tema sobre el cual deseas recibir preguntas:")
        st.session_state["user_query"] = user_query

        numero_de_preguntas = st.number_input("¿Cuántas preguntas deseas generar?", min_value=1, max_value=5, value=5)
        st.session_state["numero_de_preguntas"] = numero_de_preguntas

        if st.button("Generar preguntas"):
            preguntas_nuevas = generar_nuevas_preguntas(st.session_state["user_query"], st.session_state["numero_de_preguntas"])
            if preguntas_nuevas:
                st.session_state["preguntas"] = preguntas_nuevas
                st.session_state["respuestas"] = [None] * len(preguntas_nuevas)

    if st.session_state["preguntas"]:
        for pregunta_idx, pregunta_actual in enumerate(st.session_state["preguntas"]):
            st.write(f"Pregunta {pregunta_idx + 1}: {pregunta_actual['pregunta']}")
            st.write(f"Código:\n{pregunta_actual['codigo']}")

            opciones = [
                f"A) {pregunta_actual['opciones']['A']}",
                f"B) {pregunta_actual['opciones']['B']}",
                f"C) {pregunta_actual['opciones']['C']}",
                f"D) {pregunta_actual['opciones']['D']}"
            ]

            selected_option = st.radio(
                f"Selecciona tu respuesta para la pregunta {pregunta_idx + 1}:",
                opciones,
                index=0,
                key=f"pregunta_{pregunta_idx}"
            )

            if selected_option:
                st.session_state["respuestas"][pregunta_idx] = selected_option[0]

        if None not in st.session_state["respuestas"]:
            if st.button("Validar respuestas"):
                st.session_state["validado"] = True

        if st.session_state["validado"]:
            mostrar_resultado()

            if st.button("Reiniciar evaluación"):
                reiniciar_evaluacion()

# Ejecutar la función de mostrar
mostrar()
