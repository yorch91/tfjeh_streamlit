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
        api_key = st.secrets["OPENAI_API_KEY"]
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
def generar_nuevas_preguntas(tipo_pregunta, tema, numero_de_preguntas):
    db = cargar_documentos()

    # Plantillas para cada tipo de pregunta
    prompt_template_conceptos = """
    Eres un profesor experto en el lenguaje de programación C.
    Evita hacer referencias a C++, cin, cout, namespaces, o características específicas de C++.
    Genera {numero_de_preguntas} preguntas de opción múltiple sobre el siguiente tema: {tema}.
    Incluye opciones y la respuesta correcta en formato JSON:
    [
        {{
            "pregunta": "La pregunta generada.",
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

    prompt_template_codigo = """
    Eres un profesor experto en el lenguaje de programación C.
    Genera {numero_de_preguntas} preguntas del tipo "¿Cuál es la salida de este código?" sobre el tema: {tema}.
    Incluye el código en un bloque:
    ```
    #include <stdio.h>
    int main() {{
        // Tu código aquí
        return 0;
    }}
    ```
    Cada pregunta debe incluir opciones y una respuesta correcta en formato JSON:
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

    # Seleccionar el prompt según el tipo de pregunta
    if tipo_pregunta == "Conceptos":
        prompt_template = prompt_template_conceptos
    else:
        prompt_template = prompt_template_codigo

    prompt = PromptTemplate(
        input_variables=["context", "tema", "numero_de_preguntas"],
        template=prompt_template
    )

    try:
        docs = db.similarity_search(tema)
        context = "\n\n".join([doc.page_content for doc in docs])
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        chain = LLMChain(llm=llm, prompt=prompt)

        answer = chain.run(context=context, tema=tema, numero_de_preguntas=numero_de_preguntas)
        print("Respuesta generada:", answer)

        try:
            preguntas = json.loads(answer)
            if not isinstance(preguntas, list):
                raise ValueError("La respuesta no es una lista.")
        except json.JSONDecodeError as e:
            st.error(f"Hubo un error al procesar la respuesta. Detalles: {e}")
            return None
        except ValueError as ve:
            st.error(str(ve))
            return None

        preguntas_generadas = []
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
    st.header("Autoevaluación")
    if "preguntas" not in st.session_state:
        st.session_state["preguntas"] = []
    if "respuestas" not in st.session_state:
        st.session_state["respuestas"] = []
    if "tema" not in st.session_state:
        st.session_state["tema"] = ""
    if "tipo_pregunta" not in st.session_state:
        st.session_state["tipo_pregunta"] = "Conceptos"  # Por defecto
    if "numero_de_preguntas" not in st.session_state:
        st.session_state["numero_de_preguntas"] = 5
    if "validado" not in st.session_state:
        st.session_state["validado"] = False

    # Selección del tipo de pregunta
    tipo_pregunta = st.selectbox("Selecciona el tipo de pregunta:", ["Conceptos", "Salida de código"])
    st.session_state["tipo_pregunta"] = tipo_pregunta

    # Selección del tema
    tema = st.selectbox("Selecciona un tema:", ["Funciones", "Punteros", "Estructuras", "Control de periféricos", "Operadores", "Manejo de memoria"])
    st.session_state["tema"] = tema

    numero_de_preguntas = st.number_input("¿Cuántas preguntas deseas generar?", min_value=1, max_value=5, value=5)
    st.session_state["numero_de_preguntas"] = numero_de_preguntas

    if st.button("Generar preguntas"):
        preguntas_nuevas = generar_nuevas_preguntas(st.session_state["tipo_pregunta"], st.session_state["tema"], st.session_state["numero_de_preguntas"])
        if preguntas_nuevas:
            st.session_state["preguntas"] = preguntas_nuevas
            st.session_state["respuestas"] = [None] * len(preguntas_nuevas)

    if st.session_state["preguntas"]:
        for pregunta_idx, pregunta_actual in enumerate(st.session_state["preguntas"]):
            st.write(f"Pregunta {pregunta_idx + 1}:")
            st.write(pregunta_actual["pregunta"])

            if st.session_state["tipo_pregunta"] == "Salida de código" and "codigo" in pregunta_actual:
                st.code(pregunta_actual["codigo"], language='c')

            opciones = [
                f"A) {pregunta_actual['opciones']['A']}",
                f"B) {pregunta_actual['opciones']['B']}",
                f"C) {pregunta_actual['opciones']['C']}",
                f"D) {pregunta_actual['opciones']['D']}"
            ]

            selected_option = st.radio(f"Selecciona tu respuesta para la pregunta {pregunta_idx + 1}:", opciones, key=f"pregunta_{pregunta_idx}")

            if selected_option:
                st.session_state["respuestas"][pregunta_idx] = selected_option[0]

        if None not in st.session_state["respuestas"]:
            if st.button("Validar respuestas"):
                st.session_state["validado"] = True

        if st.session_state["validado"]:
            for pregunta_idx, pregunta_actual in enumerate(st.session_state["preguntas"]):
                st.write(f"Pregunta {pregunta_idx + 1}: {pregunta_actual['pregunta']}")
                st.write(f"Respuesta correcta: {pregunta_actual['respuesta_correcta']}")
                st.write(f"Tu respuesta: {st.session_state['respuestas'][pregunta_idx]}")

            mostrar_resultado()

            if st.button("Reiniciar evaluación"):
                reiniciar_evaluacion()

mostrar()
