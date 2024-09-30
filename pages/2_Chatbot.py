import os
import streamlit as st

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = api_key

def cargar_documentos():
    loader = TextLoader("unidades_completas.txt", encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Cargar documentos y crear embeddings
db = cargar_documentos()

# Crear el LLM de OpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Crear la cadena de pregunta-respuesta
chain = load_qa_chain(llm, chain_type="stuff")


def mostrar():

    
    st.header("Chatbot - Pregunta sobre programación en C")

    # Input del usuario para su pregunta
    user_query = st.text_input("Ingresa tu pregunta sobre programación en C:", key="user_query_input")

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



    if user_query:
        # Cargar documentos utilizando la función cargar_documentos
        db = cargar_documentos()
        # PromptTemplate que establece el rol del profesor y las restricciones
        prompt_template = """
        Eres un profesor experto en el lenguaje de programación C. 
        Responde a la pregunta proporcionada con claridad y usando únicamente el lenguaje de programación C. 
        Evita hacer referencias a C++, cin, cout, namespaces, o características específicas de C++.
        Siempre que sea posible agrega a tu explicacion codigo en C.
        Además, utiliza scanf y printf para las entradas y salidas.
        
        Documentos relacionados:
        {context}

        Pregunta: {user_query}
        """
        
        # Crear el PromptTemplate
        prompt = PromptTemplate(
            input_variables=["context", "user_query"],
            template=prompt_template
        )

        try:
            # Realizar búsqueda de documentos relevantes
            docs = db.similarity_search(user_query)

            # Combinar los documentos relevantes en un solo string como contexto
            context = "\n\n".join([doc.page_content for doc in docs])

            # Crear la cadena de pregunta-respuesta usando el template
            chain = LLMChain(llm=llm, prompt=prompt)

            # Pasar el contexto (documentos) y la consulta del usuario al LLM
            answer = chain.run(context=context, user_query=user_query)

            # Si no hay respuesta o es insuficiente, mostrar un mensaje de disculpa
            if not answer.strip():
                answer = "Lo siento, no tengo la información suficiente para responder esa pregunta. ¿Podrías intentar con otra pregunta?"

            # Formatear la respuesta con tono respetuoso y claro
            final_answer = f"Hola, aquí tienes la respuesta a tu pregunta:\n\n{answer}\n\nSi tienes más dudas, no dudes en preguntar."

            # Mostrar la respuesta
            st.write("Respuesta del profesor asistente:", final_answer)
        except Exception as e:
            st.error(f"Hubo un error al procesar tu pregunta: {e}")


mostrar()