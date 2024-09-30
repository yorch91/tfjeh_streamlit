import streamlit as st

def mostrar():
    st.header("Recursos - Documentación y tutoriales en C")
    
    st.subheader("Compiladores Online:")
    st.write("- [Replit](https://replit.com/languages/c)")
    st.write("- [JDoodle](https://www.jdoodle.com/c-online-compiler)")
    
    st.subheader("Videos tutoriales:")
    st.write("- [Curso C para principiantes - YouTube](https://www.youtube.com/watch?v=KJgsSFOSQv0)")
    
    st.subheader("Documentación sobre C:")
    st.write("- [Documentación oficial de C (en inglés)](https://en.cppreference.com/w/c)")

    st.subheader("Actividades:")
    st.write("- [Funciones](https://app.playpos.it/player_v2?type=share&bulb_id=336534&lms_launch=false)")
    st.write("- [Punteros](https://app.playpos.it/player_v2?type=share&bulb_id=336537&lms_launch=false)")
    st.write("- [Estructuras](https://app.playpos.it/player_v2?type=share&bulb_id=336551&lms_launch=false)")
    
    
    st.subheader("Videos de la catedra")
    video_url = "https://www.youtube.com/watch?v=M9ASzpVcBqk&t=53s"  # Reemplaza con tu URL
    st.video(video_url)
    video_url = "https://www.youtube.com/watch?v=sWfvziuy5Aw"  # Reemplaza con tu URL
    st.video(video_url)
    video_url = "https://youtu.be/Sdcd211gwJw?si=j44v61X3M-21LhVf"  # Reemplaza con tu URL
    st.video(video_url)
    video_url = "https://www.youtube.com/watch?v=fMJDlgJr6BY"  # Reemplaza con tu URL
    st.video(video_url)
    









mostrar()