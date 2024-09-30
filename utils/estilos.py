import streamlit as st

def aplicar_estilos():
    st.markdown(
        """
        <style>
            .css-1d391kg { background-color: #2b3e50; }
            .sidebar-button { padding: 10px 20px; background-color: #4CAF50; color: white; }
            .sidebar-button:hover { background-color: #45a049; }
        </style>
        """,
        unsafe_allow_html=True
    )
