import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# Agrega la ruta del proyecto al path de Python
root_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_path))

# Importa las funciones de autenticación
from data.EXTRAIDO.LOGIN import get_login_status

# Verificación de autenticación
if not get_login_status():
    st.error("Por favor, inicia sesión desde la página principal")
    st.stop()

# Configuración de la página
st.title("Dashboard")

# Aquí va el contenido de tu dashboard
st.write("## Resumen de Rendimiento")

# Ejemplo de carga de datos
try:
    # Reemplaza esta ruta con la ubicación real de tus datos
    data_path = Path(root_path) / "data" / "EXTRAIDO" / "datos_ejemplo.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        st.dataframe(df.head())
        
        # Ejemplo de visualización
        st.write("### Visualización de métricas clave")
        fig = px.bar(df.head(10), x='Jugador', y='Distancia')
        st.plotly_chart(fig)
    else:
        st.warning("No se encontraron datos para mostrar. Por favor, carga primero los datos.")
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")