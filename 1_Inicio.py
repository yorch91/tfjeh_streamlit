import streamlit as st
from utils.estilos import aplicar_estilos

# Aplica los estilos personalizados
aplicar_estilos()

def mostrar():
    
    st.header("Bienvenido al Tutor de Programacion en C para Electrónica : C-Companion")
    st.write("""
        Este asistente está diseñado para apoyarte en el aprendizaje y la práctica de la programación en C, con un enfoque
        en aplicaciones de electrónica. Explora los diferentes menús para interactuar con el chatbot, acceder a recursos específicos de 
        la catedra, resolver ejercicios prácticos y autoevaluarte en tiempo real
    """)

mostrar()
