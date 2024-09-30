from langchain.prompts import PromptTemplate

def obtener_prompt(tipo):
    if tipo == "chatbot":
        return PromptTemplate(
            input_variables=["context", "user_query"],
            template="""
            Eres un profesor experto en el lenguaje de programación C. 
            Responde a la pregunta proporcionada con claridad y usando únicamente el lenguaje de programación C.
            Evita hacer referencias a C++.
            Documentos relacionados: {context}

            Pregunta: {user_query}
            """
        )
    elif tipo == "evaluacion":
        return PromptTemplate(
            input_variables=["context", "user_query", "tema_aleatorio"],
            template="""
            Eres un profesor experto en el lenguaje de programación C. 
            Evita hacer referencias a C++. 
            Genera una pregunta de opción múltiple combinando el siguiente tema: {user_query} y un tema adicional: {tema_aleatorio}.
            
            Considera formular preguntas del tipo "¿Cuál es la salida de este código?".
            Si la pregunta incluye código, usa bloques delimitados de código en C.

            Documentos relacionados:
            {context}
            """
        )
