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
    # Cargar la base de datos de documentos
    db = cargar_documentos()

    # Lista de temas aleatorios
    temas_aleatorios = ["Funciones", "Punteros", "Estructuras", "Control de periféricos", "Operadores", "Manejo de memoria"]
    preguntas_generadas = []

    # Plantilla de prompt para el LLM
    prompt_template = """
    Eres un profesor experto en el lenguaje de programación C. 
    Evita hacer referencias a C++, cin, cout, namespaces, o características específicas de C++.
    Genera {numero_de_preguntas} preguntas de opción múltiple combinando el siguiente tema: {user_query} y un tema adicional: {tema_aleatorio}.
    
    También debes considerar la posibilidad de formular preguntas del tipo "¿Cuál es la salida de este código?". 
    Si optas por este tipo de pregunta, incluye el código en un bloque de código de la siguiente manera:
    ```
    #include <stdio.h>
    int main() {{
        // Tu código aquí
        return 0;
    }}
    ```
    Cada pregunta debe incluir opciones y una respuesta correcta. El formato JSON debe ser el siguiente:
    [
        {{
            "pregunta": "La pregunta generada.",
            "codigo": "Aquí va el código si la pregunta es sobre la salida del mismo.",
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

        # Combinar los documentos relevantes en un solo string como contexto
        context = "\n\n".join([doc.page_content for doc in docs])

        # Crear el LLM para generar las preguntas
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        chain = LLMChain(llm=llm, prompt=prompt)

        # Ejecutar la cadena pasando el contexto y los temas
        answer = chain.run(context=context, user_query=user_query, tema_aleatorio=random.choice(temas_aleatorios), numero_de_preguntas=numero_de_preguntas)

        # Intentar parsear la respuesta como JSON
        print("Respuesta generada:", answer)  # Imprimir la respuesta para depuración
        try:
            preguntas = json.loads(answer)  # Carga el JSON
            if not isinstance(preguntas, list):  # Verifica que sea una lista
                raise ValueError("La respuesta no es una lista.")
        except json.JSONDecodeError as e:
            st.error(f"Hubo un error al procesar tu pregunta: Respuesta no válida. Detalles: {e} Respuesta: {answer}")
            return None
        except ValueError as ve:
            st.error(str(ve))  # Manejo de valor si no es una lista
            return None

        # Asegurarse de que las preguntas sean distintas
        for pregunta in preguntas:
            if pregunta not in preguntas_generadas:
                preguntas_generadas.append(pregunta)

        return preguntas_generadas

    except Exception as e:
        st.error(f"Ocurrió un error inesperado: {e}")
        return None
    

# Función para mostrar el resultado final
def mostrar_resultado():
    correctas = sum(1 for idx, resp in enumerate(st.session_state["respuestas"]) if resp == st.session_state["preguntas"][idx]["respuesta_correcta"])
    incorrectas = len(st.session_state["respuestas"]) - correctas

    st.subheader("Resultados finales:")
    st.write(f"Respuestas correctas: {correctas}")
    st.write(f"Respuestas incorrectas: {incorrectas}")

# Función para reiniciar la evaluación
def reiniciar_evaluacion():
    st.session_state.clear()
    st.rerun()

# Función para mostrar la página de evaluación
def mostrar():
    st.header("Autoevaluacion")
    # Inicializa el estado de la sesión para almacenar preguntas, respuestas, y la opción seleccionada
    if "preguntas" not in st.session_state:
        st.session_state["preguntas"] = []
    if "respuestas" not in st.session_state:
        st.session_state["respuestas"] = []
    if "user_query" not in st.session_state:
        st.session_state["user_query"] = ""
    if "numero_de_preguntas" not in st.session_state:
        st.session_state["numero_de_preguntas"] = 5  # Inicialmente, solo se usa como valor por defecto.
    if "validado" not in st.session_state:
        st.session_state["validado"] = False

    # Entrada de la consulta del usuario solo si no hay preguntas generadas
    if not st.session_state["preguntas"]:
        user_query = st.text_input("Ingresa un tema sobre el cual deseas recibir preguntas:")
        st.session_state["user_query"] = user_query

        numero_de_preguntas = st.number_input("¿Cuántas preguntas deseas generar?", min_value=1, max_value=10, value=5)
        st.session_state["numero_de_preguntas"] = numero_de_preguntas

        if st.button("Generar preguntas"):
            preguntas_nuevas = generar_nuevas_preguntas(st.session_state["user_query"], st.session_state["numero_de_preguntas"])
            if preguntas_nuevas:
                st.session_state["preguntas"] = preguntas_nuevas
                st.session_state["respuestas"] = [None] * len(preguntas_nuevas)  # Inicializa respuestas según el número de preguntas generadas.

    # Mostrar todas las preguntas y sus opciones
    if st.session_state["preguntas"]:
        for pregunta_idx, pregunta_actual in enumerate(st.session_state["preguntas"]):
            st.write(f"Pregunta {pregunta_idx + 1}:")
            st.write(pregunta_actual["pregunta"])

            # Mostrar el código si existe, en formato de bloque de código
            if "codigo" in pregunta_actual and pregunta_actual["codigo"].strip():
                st.code(pregunta_actual["codigo"], language='c')

            # Crear una lista con las opciones formateadas
            opciones = [
                f"A) {pregunta_actual['opciones']['A']}",
                f"B) {pregunta_actual['opciones']['B']}",
                f"C) {pregunta_actual['opciones']['C']}",
                f"D) {pregunta_actual['opciones']['D']}"
            ]

            # Mostrar las opciones como radio buttons sin selección previa
            selected_option = st.radio(
                f"Selecciona tu respuesta para la pregunta {pregunta_idx + 1}:", 
                opciones, 
                index=0,
                key=f"pregunta_{pregunta_idx}"
            )

            # Guardar la respuesta seleccionada (la primera letra del string es la opción)
            if selected_option:
                st.session_state["respuestas"][pregunta_idx] = selected_option[0]

        # Habilitar el botón "Validar respuestas" solo si todas las preguntas han sido respondidas
        if None not in st.session_state["respuestas"]:
            if st.button("Validar respuestas"):
                st.session_state["validado"] = True

        # Mostrar el resultado si ya se han validado las respuestas
        if st.session_state["validado"]:
            for pregunta_idx, pregunta_actual in enumerate(st.session_state["preguntas"]):
                st.write(f"Pregunta {pregunta_idx + 1}: {pregunta_actual['pregunta']}")
                st.write(f"Respuesta correcta: {pregunta_actual['respuesta_correcta']}")
                st.write(f"Tu respuesta: {st.session_state['respuestas'][pregunta_idx]}")

            # Mostrar resultados finales
            mostrar_resultado()

            # Botón para reiniciar la evaluación
            if st.button("Reiniciar evaluación"):
                reiniciar_evaluacion()

# Ejecutar la función de mostrar
mostrar()
