import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Cargar clave API de st.secrets en Streamlit Cloud, o desde .env en local
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = os.getenv('OPENAI_API_KEY')

# Verificar si la clave de API está presente
if api_key is None:
    raise ValueError("No se encontró 'OPENAI_API_KEY'. Asegúrate de que esté definida correctamente en los secretos o en .env.")

# Asignar la clave a las variables de entorno
os.environ["OPENAI_API_KEY"] = api_key

def cargar_documentos():
    loader = TextLoader("unidades_completas.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Cargar documentos y crear embeddings
db = cargar_documentos()

# Crear el LLM de OpenAI
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Inicializar una variable de sesión para el historial de conversación
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def agregar_a_contexto(user_query, respuesta):
    """Agrega la pregunta y respuesta más recientes al historial de conversación."""
    st.session_state['chat_history'].append({"query": user_query, "answer": respuesta})

def obtener_contexto_limited():
    """Recuperar la última interacción para reducir el tamaño de tokens."""
    if len(st.session_state['chat_history']) > 1:
        return [st.session_state['chat_history'][-1]]  # Solo última interacción
    else:
        return st.session_state['chat_history']

def construir_contexto():
    """Construir un contexto compacto basado en la última interacción."""
    historial = obtener_contexto_limited()
    contexto = ""
    for interaction in historial:
        contexto += f"Pregunta: {interaction['query']}\nRespuesta: {interaction['answer']}\n"
    return contexto

def mostrar():
    st.header("Chatbot - Pregunta sobre programación en C")

    # Seleccionar el nivel de experiencia
    level_of_experience = st.selectbox("Selecciona tu nivel de experiencia:", 
                                       options=["Nada experimentado", "Poco experimentado", "Experimentado"])

    # Mensaje personalizado según el nivel de experiencia
    explanation_modifier = {
        "Nada experimentado": "Proporcione una explicación sencilla con ejemplos claros.",
        "Poco experimentado": "Explica de forma moderada con algunos detalles técnicos.",
        "Experimentado": "Asume conocimiento intermedio de C, proporciona respuestas concisas."
    }

    # Input del usuario para su pregunta
    user_query = st.text_input("Ingresa tu pregunta sobre programación en C:", key="user_query_input")

    # Botón para realizar la consulta después de validar el input
    if st.button("Realizar consulta"):
        # Validar que el input no esté vacío y no exceda los 50 caracteres
        if not user_query:
            st.warning("El campo no puede estar vacío. Por favor, ingresa una pregunta.")
        elif len(user_query) > 50:
            st.warning("La pregunta no debe exceder los 50 caracteres.")
        else:
            # Continuar con la ejecución si la validación es exitosa
            # Crear el contexto con las últimas interacciones
            conversation_history = construir_contexto()

            # PromptTemplate que establece el rol del profesor y las restricciones
            prompt_template = """
            Eres un experto en C. Responde solo usando C, sin mencionar C++ ni características relacionadas.
            {experience_level}

            Documentos relevantes:
            {context}

            Pregunta: {user_query}
            """

            # Crear el PromptTemplate
            prompt = PromptTemplate(
                input_variables=["context", "user_query", "experience_level"],
                template=prompt_template
            )

            try:
                # Realizar búsqueda de documentos relevantes, usando solo 500 caracteres de cada documento
                docs = db.similarity_search(user_query)

                # Combinar los documentos relevantes en un solo string como contexto (solo hasta 2 documentos para reducir tokens)
                context = "\n\n".join([doc.page_content[:500] for doc in docs[:2]])

                # Crear la cadena de pregunta-respuesta usando el template
                chain = LLMChain(llm=llm, prompt=prompt)

                # Pasar el contexto (documentos), la consulta del usuario y el nivel de experiencia al LLM
                answer = chain.run(context=context, user_query=user_query, 
                                   experience_level=explanation_modifier[level_of_experience])

                # Si no hay respuesta o es insuficiente, mostrar un mensaje de disculpa
                if not answer.strip():
                    answer = "Lo siento, no tengo la información suficiente para responder esa pregunta. ¿Podrías intentar con otra pregunta?"

                # Formatear la respuesta con tono respetuoso y claro
                final_answer = f"Hola, aquí tienes la respuesta a tu pregunta:\n\n{answer}\n\nSi tienes más dudas, no dudes en preguntar."

                # Guardar el historial de conversación
                agregar_a_contexto(user_query, final_answer)

                # Mostrar la respuesta
                st.write("Respuesta del profesor asistente:", final_answer)
            except Exception as e:
                st.error(f"Hubo un error al procesar tu pregunta: {e}")

    # Mostrar el historial de conversación en un desplegable
    if st.session_state.chat_history:
        with st.expander("Historial de conversación"):
            for i, interaction in enumerate(st.session_state.chat_history):
                st.write(f"**Consulta {i+1}:** {interaction['query']}")
                st.write(f"**Respuesta {i+1}:** {interaction['answer']}")

    # Nuevo input para generar prompts
    user_topic = st.text_input("Ingresa un tema para generar prompts:", key="user_topic_input")

    # Botón para generar prompts
    if st.button("Generar Prompt"):
        if user_topic:
            # Sugerencias de prompts bien formateados
            posibles_prompts = [
                f"Explica cómo funcionan los {user_topic} en C.",
                f"¿Cómo se declaran y usan los {user_topic} en C?",
                f"¿Cuáles son los errores comunes al trabajar con {user_topic} en C?",
                f"Muestra un ejemplo de código en C utilizando {user_topic}.",
                f"¿Cómo se relacionan los {user_topic} con la memoria en C?"
            ]

            st.write("Sugerencias de prompts:")
            for prompt in posibles_prompts:
                st.code(prompt)
        else:
            st.warning("Por favor, ingresa un tema para generar prompts.")

mostrar()
