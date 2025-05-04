import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np

# Agregar la ruta del proyecto al path de Python
root_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_path))

# Importar módulos de autenticación
from auth.session import initialize_session, get_login_status, set_login_status
from auth.login import login_form, logout

# Configuración inicial de la página
st.set_page_config(
    page_title="Análisis Deportivo",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Ocultar las páginas automáticas
css = '''
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none}
    div[data-testid="stSidebarNav"] {display: none;}
'''
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Inicializar la sesión
initialize_session()

# Función principal
def main():
    st.title("Aplicación de Análisis Deportivo")

    if login_form():
        # Personalización de la barra lateral
        with st.sidebar:
            # Encabezado con información del sistema
            st.title('Análisis Univesitario')
            st.subheader(f"Bienvenido, Usuario")
            
            # Mostrar logo en la barra lateral
            try:
                # Usar la ruta correcta para la imagen
                logo_path = Path(root_path, "data", "escudo uni.jpg")
                if logo_path.exists():
                    logo = Image.open(logo_path)
                    st.image(logo, width=150)
                else:
                    st.warning("Logo no encontrado en: " + str(logo_path))
            except Exception as e:
                st.warning(f"No se pudo cargar el logo: {e}")
            
            with st.container():
                st.write("🎯 **Metricas**")
                if Path(f"{Path(__file__).parent}/pages/Dashoboard.py").exists():
                    st.page_link("pages/Dashoboard.py", label=" Dashboard Principal", icon="📊")
                if Path(f"{Path(__file__).parent}/pages/Equipeeee.py").exists():
                    st.page_link("pages/Equipeeee.py", label=" Análisis de Equipo", icon="⚽")
                if Path(f"{Path(__file__).parent}/pages/Jugadorr.py").exists():
                    st.page_link("pages/Jugadorr.py", label=" Análisis de Jugador", icon="👤")

            # Información del sistema
            st.divider()
            with st.expander("ℹ️ Información del Sistema"):
                st.write("**Versión del Sistema:** 1.0.0")
                st.write("**Última Actualización:** 2024-05-02")
                st.write("**Soporte:** calvoj550@gmail.com")
                st.write("**Documentación:** [Ver Manual de Usuario]()")

            # Footer
            st.divider()
            st.caption("© 2025 Sistema de Análisis Deportivo - Todos los derechos reservados")
            
            # Botón de cierre de sesión
            if st.button("🚪 Cerrar Sesión", type="primary"):
                logout()
                st.rerun()

        # Contenido principal
        st.write("### Bienvenido al sistema de análisis deportivo")
        
        # Mensaje sobre próximas actualizaciones
        st.info("""
        🔔 **Próximamente:** 
        Estamos trabajando en la incorporación de nuevas funcionalidades y visualizaciones avanzadas 
        que te permitirán analizar el rendimiento del equipo con mayor profundidad. 
        Muy pronto encontrarás aquí gráficos interactivos, análisis comparativos y métricas personalizadas.
        """)
        
        # Resumen en columnas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Distancia Total (equipo)", value="127.5 km", delta="2.3 km")
        with col2:
            st.metric(label="Sprints", value="87", delta="-3")
        with col3:
            st.metric(label="Aceleraciones", value="245", delta="15")
        
        # Información general
        st.subheader("Información general")
        st.write("""
        Este dashboard proporciona una visión general del rendimiento del equipo y los jugadores. 
        Utiliza la barra lateral para navegar a secciones específicas de la aplicación.
        """)
        
        # Datos de ejemplo
        st.subheader("Últimos datos registrados")
        data = {
            "Fecha": ["01/05/2024", "28/04/2024", "25/04/2024", "22/04/2024"],
            "Sesión": ["Partido", "Entrenamiento", "Partido", "Entrenamiento"],
            "Duración": ["90 min", "120 min", "90 min", "110 min"],
            "Dist. Total": ["127.5 km", "98.2 km", "125.7 km", "92.5 km"]
        }
        st.dataframe(data, use_container_width=True)
    else:
        st.write("Por favor, inicia sesión para acceder al contenido.")

if __name__ == "__main__":
    main()