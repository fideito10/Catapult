# auth/session.py
import streamlit as st
import time
import hashlib
from pathlib import Path

def get_login_status():
    """Obtiene el estado de inicio de sesión actual"""
    return st.session_state.get('logged_in', False)

def set_login_status(status):
    """Establece el estado de inicio de sesión"""
    st.session_state['logged_in'] = status

def generate_token():
    """Genera un token único para la sesión"""
    return hashlib.sha256(str(time.time()).encode()).hexdigest()

def check_credentials(password):
    """Verifica las credenciales del usuario"""
    return password == "universitario"

def initialize_session():
    """Inicializa la sesión si existe un token válido"""
    # Actualizar la API para obtener parámetros de consulta
    query_params = st.experimental_get_query_params()
    if 'session_token' in query_params and not get_login_status():
        set_login_status(True)