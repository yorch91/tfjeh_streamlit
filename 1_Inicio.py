import streamlit as st
from utils.estilos import aplicar_estilos

# Aplica los estilos personalizados
aplicar_estilos()

def mostrar():
    
    st.header("Bienvenido al Profesor Asistente de Programación en C")
    st.write("""
        Este asistente está diseñado para ayudarte a aprender y practicar programación en C. 
        Explora los diferentes menús para interactuar con el chatbot, acceder a recursos, 
        practicar con ejercicios o autoevaluarte.
    """)

mostrar()