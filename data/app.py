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
root_path = Path(__file__).parent.absolute()
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
    st.title("Sistema de Análisis para Rugby")
    st.markdown("""
    # Bienvenido al Sistema de Análisis

    Esta herramienta proporciona diferentes módulos para analizar el rendimiento y 
    estado físico de los jugadores de rugby.

    Selecciona una opción del menú lateral para comenzar.
    """)

    if login_form():
        # Personalización de la barra lateral
        with st.sidebar:
            # Encabezado con información del sistema
            st.title('Análisis Univesitario')
            st.subheader(f"Bienvenido, Usuario")
            
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
                st.write("**Documentación:** [Ver Manual de Usuario](https://github.com/usuario/proyecto/docs)")

            # Footer
            st.divider()
            st.caption("© 2025 Sistema de Análisis Deportivo - Todos los derechos reservados")
            
            # Botón de cierre de sesión
            if st.button("🚪 Cerrar Sesión", type="primary"):
                logout()
                st.rerun()

        # Contenido principal
        
        # Introducción a la aplicación
        st.header("Sistema de Análisis Deportivo")
        
        st.subheader("¿Cómo funciona esta aplicación?")
        st.write("""
        Esta plataforma te permite analizar datos deportivos recopilados durante entrenamientos y partidos.
        Diseñada específicamente para el Club Universitario de la Plata, ofrece métricas
        detalladas que ayudan a optimizar el rendimiento de los jugadores y del equipo.
        """)
        
        # Información sobre las secciones
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("### 📊 Análisis de Equipo")
            st.write("""
            En la sección de **Análisis de Equipo** podrás:
            
            - Visualizar métricas colectivas como distancia total recorrida
            - Analizar tendencias de rendimiento del equipo a lo largo del tiempo
            - Comparar estadísticas entre diferentes partidos y entrenamientos
            - Identificar patrones tácticos y áreas de mejora grupal
            """)
        
        with col2:
            st.info("### 👤 Análisis de Jugador")
            st.write("""
            En la sección de **Análisis de Jugador** podrás:
            
            - Examinar métricas individuales como sprints y aceleraciones
            - Evaluar el progreso de cada atleta durante la temporada
            - Comparar rendimiento entre jugadores de la misma posición
            - Identificar fortalezas y áreas de desarrollo personales
            """)
        
    
        
        
    else:
        st.write("Por favor, inicia sesión para acceder al contenido.")

if __name__ == "__main__":
    main()