# Importaciones estándar
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import os

# Importaciones para visualización
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Análisis Nutricional",
    page_icon=":rugby_football:",
    layout="wide"
)

# Título y descripción
st.title("Análisis Nutricional y Composición Corporal")
st.markdown("""
Esta herramienta analiza la evolución de la composición corporal de los jugadores de rugby,
ofreciendo interpretaciones y recomendaciones personalizadas.
""")


def cargar_datos_google_sheets(sheet_url=None):
    """
    Carga datos desde una hoja de Google Sheets y los procesa para su análisis.
    
    Args:
        sheet_url (str, optional): URL de la hoja de Google Sheets. 
            Si es None, se usa una URL predeterminada.
            
    Returns:
        pd.DataFrame: DataFrame con los datos cargados desde Google Sheets
    """
    if sheet_url is None:
        # URL de la hoja de cálculo compartida por defecto
        sheet_url = "https://docs.google.com/spreadsheets/d/17rtbwMwGMXvoE4sXdGNXXr19z73mdCSjxTApoG5v_6o/edit?gid=0#gid=0"
    
    # Extraer el ID de la hoja de cálculo de la URL
    try:
        if 'spreadsheets/d/' in sheet_url:
            sheet_id = sheet_url.split('spreadsheets/d/')[1].split('/')[0]
        else:
            # Si no tiene el formato esperado
            raise ValueError("La URL no parece ser una URL válida de Google Sheets")
        
        # Construir la URL para exportar como CSV
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        
        # Cargar los datos desde la URL
        st.write("📊 Cargando datos desde Google Sheets...")
        df = pd.read_csv(csv_url)
        
        # Mostrar información sobre los datos cargados en el log
        st.write(f"✅ Datos cargados exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas")
        
        # Procesar las columnas de fecha si existen
        if 'Fecha' in df.columns:
            try:
                df['fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y')
                df['fecha_str'] = df['fecha'].dt.strftime('%d/%m/%Y')
                st.write("✅ Columna de fecha procesada correctamente")
            except Exception as e:
                st.warning(f"⚠️ No se pudo procesar la columna de fecha: {e}")
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return pd.DataFrame()  # Devolver un DataFrame vacío en caso de error


def normalizar_valores_nan(dataframe):
    """
    Normaliza el DataFrame reemplazando los valores NaN por 0.
    
    Args:
        dataframe (pd.DataFrame): El DataFrame a normalizar
        
    Returns:
        pd.DataFrame: DataFrame con valores NaN reemplazados por 0
    """
    try:
        # Crear una copia para no modificar el original directamente
        df_normalizado = dataframe.copy()
        
        # Reemplazar todos los valores NaN por 0
        df_normalizado = df_normalizado.fillna(0)
        
        # Contar cuántos valores fueron reemplazados
        total_nan = dataframe.isna().sum().sum()
        st.write(f"✅ Se han normalizado {total_nan} valores NaN a 0 en el DataFrame")
        
        # Mostrar un resumen de las columnas afectadas
        columnas_afectadas = dataframe.columns[dataframe.isna().any()].tolist()
        if columnas_afectadas and total_nan > 0:
            with st.expander("Ver detalles de normalización"):
                st.write("Columnas normalizadas:")
                for col in columnas_afectadas:
                    num_nan = dataframe[col].isna().sum()
                    st.write(f"- {col}: {num_nan} valores reemplazados")
                
        return df_normalizado
    except Exception as e:
        st.error(f"❌ Error al normalizar valores: {e}")
        return dataframe  # Devolver el DataFrame original en caso de error


def seleccionar_jugador(dataframe):
    """
    Permite seleccionar interactivamente un jugador del DataFrame y muestra su información.
    
    Args:
        dataframe (pd.DataFrame): DataFrame con los datos de los jugadores
    
    Returns:
        dict: Datos del jugador seleccionado o None si hay error
    """
    try:
        # Determinar qué columna usar para identificar jugadores
        columna_jugador = None
        for col in ['Apellido', 'Nombre', 'Jugador', 'Nombre Completo']:
            if col in dataframe.columns:
                columna_jugador = col
                break
        
        if columna_jugador is None:
            st.error("❌ Error: No se encontró ninguna columna con nombres de jugadores en el DataFrame")
            st.write("Columnas disponibles:", ", ".join(dataframe.columns))
            return None
        
        # Obtener la lista de jugadores únicos y ordenarlos alfabéticamente
        jugadores = sorted(dataframe[columna_jugador].dropna().unique())
        
        if len(jugadores) == 0:
            st.error("❌ No hay jugadores en el DataFrame")
            return None
        
        # Crear título para la sección
        st.subheader("🏉 Selecciona un jugador para ver su información:")
        
        # Crear un dropdown para seleccionar jugadores con Streamlit
        jugador_seleccionado = st.selectbox(
            label="Jugador",
            options=jugadores,
            index=0,
            key="selector_jugador"
        )
        
        # Si hay un jugador seleccionado, mostrar su información
        if jugador_seleccionado:
            # Filtrar datos del jugador
            datos_jugador = dataframe[dataframe[columna_jugador] == jugador_seleccionado]
            
            if not datos_jugador.empty:
                # Crear una sección para mostrar la información
                with st.container():
                    mostrar_info_jugador(dataframe, jugador_seleccionado)
                    # Devolver los datos del jugador seleccionado
                    return datos_jugador.iloc[0].to_dict()
            else:
                st.warning(f"⚠️ No se encontraron datos para el jugador: {jugador_seleccionado}")
                return None
        
    except Exception as e:
        st.error(f"❌ Error al crear el selector de jugadores: {str(e)}")
        st.exception(e)  # Muestra el traceback completo para depuración
        return None
    
    
    
def mostrar_info_jugador(dataframe, apellido):
    """
    Muestra un dashboard con información personal del jugador seleccionado.
    
    Args:
        dataframe (pd.DataFrame): DataFrame con los datos de los jugadores
        apellido (str): Apellido del jugador a buscar
    """
    try:
        # Buscar el jugador en el DataFrame (ignorando mayúsculas/minúsculas)
        jugador = dataframe[dataframe['Apellido'].str.contains(apellido, case=False, na=False)]
        
        if len(jugador) == 0:
            st.error(f"❌ No se encontró ningún jugador con el apellido '{apellido}'")
            return
        
        # Tomar el primer registro si hay múltiples coincidencias
        jugador = jugador.iloc[0]
        
        # Crear el dashboard con componentes de Streamlit
        st.divider()
        st.header("🏉 DASHBOARD DE COMPOSICIÓN CORPORAL - JUGADOR DE RUGBY")
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧍‍♂️ INFORMACIÓN PERSONAL")
            st.markdown(f"**👤 Nombre:** {jugador['Apellido']}")
            
            # Verificar si existe cada columna antes de mostrarla
            if 'Posición' in jugador:
                st.markdown(f"**🏈 Posición:** {jugador['Posición']}")
            elif 'Posicion' in jugador:  # Alternativa sin tilde
                st.markdown(f"**🏈 Posición:** {jugador['Posicion']}")
            
            if 'Talla (cm)' in jugador:
                st.markdown(f"**📏 Altura:** {jugador['Talla (cm)']} cm")
            elif 'Altura' in jugador:
                st.markdown(f"**📏 Altura:** {jugador['Altura']} cm")
            
            if 'Peso (kg)' in jugador:
                st.markdown(f"**⚖️ Peso:** {jugador['Peso (kg)']} kg")
            elif 'Peso' in jugador:
                st.markdown(f"**⚖️ Peso:** {jugador['Peso']} kg")
            
            if 'fecha_str' in jugador:
                st.markdown(f"**📅 Última medición:** {jugador['fecha_str']}")
            
            if 'Objetivo 1' in jugador:
                st.markdown(f"**🎯 Objetivo:** {jugador['Objetivo 1']}")
            elif 'Objetivo' in jugador:
                st.markdown(f"**🎯 Objetivo:** {jugador['Objetivo']}")
                
        # En la segunda columna podemos añadir gráficos o información adicional
        with col2:
            # Aquí puedes añadir gráficos o más información si lo deseas
            pass
            
        st.divider()
        
    except Exception as e:
        st.error(f"❌ Error al procesar la información: {e}")
        
        # Mostrar las columnas disponibles para ayudar a depurar
        st.write("Columnas disponibles en el DataFrame:")
        for i, col in enumerate(dataframe.columns):
            st.write(f"{i+1}. {col}")
        
    return








# Sección principal de la aplicación
def main():
    """
    Función principal que ejecuta la aplicación Streamlit
    """
    # Cargar datos desde Google Sheets
    with st.spinner("Cargando datos..."):
        df = cargar_datos_google_sheets()
    
    # Normalizar valores NaN si es necesario
    if not df.empty:
        df = normalizar_valores_nan(df)
        st.success(f"Datos cargados correctamente: {df.shape[0]} filas y {df.shape[1]} columnas")
        
        # Sección para visualizar información de jugadores
        st.header("Análisis Individual de Jugadores")
        st.write("En esta sección puedes ver información detallada de cada jugador.")
        
        # Agregar selección de jugador
        jugador_datos = seleccionar_jugador(df)

        # Verificar si se obtuvo información del jugador
        if jugador_datos:
            st.success(f"Jugador seleccionado correctamente: {jugador_datos.get('Apellido', 'Nombre no disponible')}")
            
            # Aquí puedes agregar más análisis específicos para el jugador seleccionado
            st.subheader("Análisis personalizado")
            st.write("Aquí irán los análisis específicos para el jugador seleccionado.")
            
            # Puedes acceder a los datos así: jugador_datos.get('nombre_columna', 'valor_por_defecto')
        else:
            st.warning("No se ha podido cargar la información del jugador. Por favor, seleccione otro jugador.")# Ejecutar la aplicación
if __name__ == "__main__":
    main()
