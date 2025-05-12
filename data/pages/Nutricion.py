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
    Carga datos desde Google Sheets a través de su URL compartida.
    
    Args:
        sheet_url (str, opcional): URL de la hoja de Google Sheets. 
                                   Si es None, usa una URL predeterminada.
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    if sheet_url is None:
        # URL de la hoja de cálculo compartida por defecto
        sheet_url = "https://docs.google.com/spreadsheets/d/17rtbwMwGMXvoE4sXdGNXXr19z73mdCSjxTApoG5v_6o/edit?gid=0#gid=0"
    
    with st.spinner("Cargando datos..."):
        try:
            # Extraer el ID de la hoja de cálculo de la URL
            if 'spreadsheets/d/' in sheet_url:
                sheet_id = sheet_url.split('spreadsheets/d/')[1].split('/')[0]
            else:
                st.error("La URL no parece ser una URL válida de Google Sheets")
                raise ValueError("URL de Google Sheets inválida")
            
            # Construir la URL para exportar como CSV
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            
            # Leer los datos directamente desde la URL
            df = pd.read_csv(csv_url)
            st.success(f"Datos cargados exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas")
            
            # Convertir fechas si hay una columna de fecha
            if 'fecha' in df.columns:
                # Lista de formatos a probar para mayor flexibilidad
                formatos = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%y', '%d/%m/%y']
                
                # Intentar cada formato hasta que uno funcione
                fecha_convertida = False
                for formato in formatos:
                    try:
                        # Intentar convertir con el formato actual
                        df['fecha'] = pd.to_datetime(df['fecha'], format=formato, errors='coerce')
                        
                        # Si no hay fechas NaN, hemos encontrado el formato correcto
                        if not df['fecha'].isna().any():
                            fecha_convertida = True
                            st.info(f"✅ Fechas convertidas correctamente usando el formato: {formato}")
                            break
                    except:
                        continue
                
                # Si aún hay fechas nulas después de intentar todos los formatos
                if df['fecha'].isna().any():
                    # Detectar filas con problemas
                    filas_problematicas = df[df['fecha'].isna()]
                    st.warning(f"⚠️ Algunas fechas no pudieron ser convertidas. Verificar que el formato sea día-mes-año (ej: 3-2-2025)")
                    if len(filas_problematicas) > 0:
                        st.warning(f"⚠️ Ejemplo de valor problemático: {filas_problematicas['fecha'].iloc[0]}")
                
                # Crear columna con formato legible
                df['fecha_str'] = df['fecha'].dt.strftime('%d/%m/%Y')
            else:
                st.warning("⚠️ No se encontró la columna 'fecha' en los datos. Verificar estructura del Google Sheet.")
            
            return df
        
        except Exception as e:
            st.error(f"Error al cargar datos: {str(e)}")
            # Proporcionar datos de ejemplo en caso de error
            st.info("Generando datos de ejemplo para demostración...")
            return generar_datos_ejemplo()

def generar_datos_ejemplo():
    """
    Genera un conjunto de datos de ejemplo para demostración.
    
    Returns:
        pd.DataFrame: DataFrame con datos de ejemplo
    """
    fechas = pd.date_range(start='2024-01-01', periods=6, freq='MS')
    datos = {
        'fecha': fechas,
        'Apellido': ['Jugador Demo'] * 6,
        'Peso (kg)': [95.2, 94.8, 93.5, 92.9, 92.3, 91.8],
        'Talla (cm)': [188] * 6,
        'masa_muscular': [42.5, 42.8, 43.2, 43.5, 43.7, 43.9],
        'masa_adiposa': [24.3, 23.5, 22.1, 21.2, 20.5, 19.8],
        'masa_osea': [15.1, 15.1, 15.1, 15.2, 15.2, 15.2],
        'pliegue_triceps': [15.2, 14.8, 14.1, 13.7, 13.2, 12.8],
        'pliegue_subescapular': [16.8, 16.3, 15.7, 15.2, 14.8, 14.3],
        'pliegue_suprailiaco': [18.2, 17.5, 16.8, 16.2, 15.6, 15.1],
        'pliegue_abdominal': [22.5, 21.8, 20.5, 19.8, 19.2, 18.5],
        'pliegue_muslo': [15.7, 15.2, 14.6, 14.2, 13.8, 13.5],
        'pliegue_pantorrilla': [10.2, 9.8, 9.5, 9.2, 8.9, 8.7],
        'Posición': ['Ala'] * 6,
        'Objetivo 1': ['Aumentar masa muscular'] * 6
    }
    df = pd.DataFrame(datos)
    df['fecha_str'] = df['fecha'].dt.strftime('%d/%m/%Y')
    return df

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
        if total_nan > 0:
            st.info(f"✅ Se han normalizado {total_nan} valores faltantes a 0 en el DataFrame")
            
            # Mostrar un resumen de las columnas afectadas
            columnas_afectadas = dataframe.columns[dataframe.isna().any()].tolist()
            if columnas_afectadas:
                with st.expander("Ver detalles de valores normalizados"):
                    for col in columnas_afectadas:
                        num_nan = dataframe[col].isna().sum()
                        st.text(f"- {col}: {num_nan} valores reemplazados")
                
        return df_normalizado
    except Exception as e:
        st.error(f"❌ Error al normalizar valores: {str(e)}")
        return dataframe  # Devolver el DataFrame original en caso de error

def mostrar_info_jugador(df, apellido):
    """
    Muestra un dashboard con información personal del jugador seleccionado.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de todos los jugadores
        apellido (str): Apellido del jugador a buscar
    """
    try:
        # Buscar el jugador en el DataFrame (ignorando mayúsculas/minúsculas)
        jugador = df[df['Apellido'].str.contains(apellido, case=False, na=False)]
        
        if len(jugador) == 0:
            st.warning(f"No se encontró ningún jugador con el apellido '{apellido}'")
            return
        
        # Tomar el primer registro si hay múltiples coincidencias
        jugador = jugador.iloc[0]
        
        # Crear el dashboard usando componentes de Streamlit
        st.subheader("🏉 DASHBOARD DE COMPOSICIÓN CORPORAL")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧍‍♂️ Información Personal")
            info_personal = {
                "👤 Nombre": jugador.get('Apellido', 'No disponible'),
                "🏈 Posición": jugador.get('Posición', jugador.get('Posicion', 'No disponible')),
                "📏 Altura": f"{jugador.get('Talla (cm)', jugador.get('Altura', 'No disponible'))} cm",
                "⚖️ Peso": f"{jugador.get('Peso (kg)', jugador.get('Peso', 'No disponible'))} kg",
                "📅 Última medición": jugador.get('fecha_str', 'No disponible'),
                "🎯 Objetivo": jugador.get('Objetivo 1', jugador.get('Objetivo', 'No disponible'))
            }
            
            for clave, valor in info_personal.items():
                st.markdown(f"**{clave}:** {valor}")
        
        with col2:
            # Aquí podrías añadir visualizaciones o más información
            st.markdown("### 📊 Indicadores Principales")
            
            # Ejemplo de métricas
            if 'Peso (kg)' in jugador and 'masa_muscular' in jugador and 'masa_adiposa' in jugador:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Peso (kg)", jugador.get('Peso (kg)', 0))
                with col_b:
                    st.metric("Masa Muscular", jugador.get('masa_muscular', 0))
                with col_c:
                    st.metric("Masa Adiposa", jugador.get('masa_adiposa', 0))
        
    except Exception as e:
        st.error(f"Error al procesar la información: {str(e)}")
        
        # Mostrar las columnas disponibles para ayudar a depurar
        with st.expander("Columnas disponibles para depuración"):
            for i, col in enumerate(df.columns):
                st.text(f"{i+1}. {col}")
    
    return

# Sección principal de la aplicación
def main():
    # Sidebar para opciones
    st.sidebar.title("Opciones")
    sheet_url = st.sidebar.text_input(
        "URL de Google Sheets (opcional)",
        "https://docs.google.com/spreadsheets/d/17rtbwMwGMXvoE4sXdGNXXr19z73mdCSjxTApoG5v_6o/edit?gid=0#gid=0"
    )
    
    # Cargar datos
    df = cargar_datos_google_sheets(sheet_url if sheet_url else None)
    
    # Normalizar valores NaN
    df = normalizar_valores_nan(df)
    
    # Selector de jugador
    if 'Apellido' in df.columns:
        jugadores = sorted(df['Apellido'].unique())
        jugador_seleccionado = st.sidebar.selectbox("Seleccionar Jugador", jugadores)
        
        if st.sidebar.button("Ver Información"):
            mostrar_info_jugador(df, jugador_seleccionado)
    else:
        st.warning("No se encontró la columna 'Apellido' en los datos")
    
    # Mostrar datos en bruto en un expander (opcional)
    with st.expander("Ver datos en bruto"):
        st.dataframe(df)

if __name__ == "__main__":
    main()