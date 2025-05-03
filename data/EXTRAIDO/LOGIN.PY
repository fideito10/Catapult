import streamlit as st
import time
import hashlib

def get_login_status():
    return st.session_state.get('logged_in', False)

def set_login_status(status):
    st.session_state['logged_in'] = status

def generate_token():
    return hashlib.sha256(str(time.time()).encode()).hexdigest()

def check_password():
    """Verifica la contraseña y maneja el estado de la sesión."""
    if get_login_status():
        return True

    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar Sesión"):
        if password == "universitario":
            token = generate_token()
            set_login_status(True)
            st.query_params.from_dict({"session_token": token})
            st.rerun()
        else:
            st.error("😕 Contraseña incorrecta")
    
    return False

def logout():
    """Cierra la sesión del usuario."""
    set_login_status(False)
    st.query_params.clear()  # Limpia todos los parámetros de la URL
    st.rerun()

# Aplicación principal
def main():
    st.title("Aplicación de Análisis Deportivo")

    # Verificar el token de sesión en la URL
    params = st.query_params.to_dict()
    if 'session_token' in params and not get_login_status():
        set_login_status(True)

    if check_password():
        st.write("Bienvenido al análisis deportivo!")
        
        # Aquí va el contenido principal de tu aplicación
        st.write("Contenido protegido de la aplicación...")

        # Botón de cierre de sesión
        if st.button("Cerrar Sesión"):
            logout()
    else:
        st.write("Por favor, inicia sesión para acceder al contenido.")

if __name__ == "__main__":
    main()
