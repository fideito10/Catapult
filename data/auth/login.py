# auth/login.py
import streamlit as st
from pathlib import Path
from .session import get_login_status, set_login_status, generate_token, check_credentials

def login_form():
    """Muestra el formulario de inicio de sesión"""
    if get_login_status():
        return True

    with st.form("login_form"):
        password = st.text_input("Contraseña", type="password")
        submit = st.form_submit_button("Iniciar Sesión")
        
        if submit:
            if check_credentials(password):
                token = generate_token()
                set_login_status(True)                # Actualizar la API para manipular parámetros de consulta
                st.experimental_set_query_params(session_token=token)
                st.rerun()
                return True
            else:
                st.error("😕 Contraseña incorrecta")
    return False

def logout():
    """Cierra la sesión del usuario"""
    set_login_status(False)
    # Actualizar la API para limpiar parámetros de consulta
    st.experimental_set_query_params()
    st.rerun()

def require_auth():
    """Verifica autenticación y redirecciona si no está autenticado"""
    if not get_login_status():
        st.error("Por favor, inicia sesión desde la página principal")
        st.stop()