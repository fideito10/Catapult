import sys
from pathlib import Path

# Agregar la ruta del proyecto al path de Python
root_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_path))

# Primero configuramos la página antes de cualquier otra llamada a Streamlit
import streamlit as st
st.set_page_config(page_title="Universitario de La Plata - Dashboard Jugador", layout="wide")

# Ahora importamos el resto de las librerías
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

# Importar el módulo de autenticación
try:
    from auth.login import require_auth
    from auth.session import initialize_session
    
    # Inicializar y verificar autenticación
    initialize_session()
    require_auth()
except ImportError:
    st.error("No se pudo importar el módulo de autenticación. Asegúrate de que exista la carpeta 'auth' con los archivos necesarios.")
    st.stop()

# Ocultar las páginas automáticas
css = '''
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none}
    div[data-testid="stSidebarNav"] {display: none;}
'''
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Función para agregar logo y título
def add_header():
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.title("Universitario de La Plata")
        st.markdown("### Dashboard de Rendimiento Individual")
    
    with col2:
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

def obtener_valores_maximos_por_jugador(dataframe, columnas, jugador=None, excluir_aceleraciones=True):
    """
    Obtiene los valores máximos históricos por jugador para las columnas especificadas.
    
    Parámetros:
    - dataframe: DataFrame con los datos de Catapult
    - columnas: Lista de nombres de columnas para obtener sus valores máximos
    - jugador: (Opcional) Nombre del jugador específico. Si es None, devuelve para todos los jugadores
    - excluir_aceleraciones: (Opcional) Si es True, excluye columnas de aceleraciones y desaceleraciones
    
    Retorna:
    - DataFrame con los valores máximos de cada columna para cada jugador
    """
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()
    
    # Verificar que 'Player Name' está en las columnas
    if 'Player Name' not in dataframe.columns:
        print("Error: No se encuentra la columna 'Player Name' en el DataFrame")
        return pd.DataFrame()
    
    # Filtrar aceleraciones y desaceleraciones si se solicita
    if excluir_aceleraciones:
        columnas = [col for col in columnas if 'cceleration' not in col and 'eceleration' not in col]
    
    # Verificar que todas las columnas solicitadas existen
    columnas_validas = [col for col in columnas if col in dataframe.columns]
    if len(columnas_validas) < len(columnas):
        columnas_faltantes = set(columnas) - set(columnas_validas)
        print(f"Advertencia: No se encontraron estas columnas: {columnas_faltantes}")
    
    # Filtrar por jugador específico si se proporciona
    if jugador:
        df_filtrado = dataframe[dataframe['Player Name'] == jugador].copy()
        if df_filtrado.empty:
            print(f"No se encontraron datos para el jugador: {jugador}")
            return pd.DataFrame()
    else:
        df_filtrado = dataframe.copy()
    
    # Convertir columnas numéricas (manejo de posibles formatos de texto)
    for col in columnas_validas:
        if df_filtrado[col].dtype == 'object':
            # Reemplazar comas por puntos si es necesario
            df_filtrado[col] = df_filtrado[col].str.replace(',', '.', regex=True)
        # Convertir a numérico con manejo de errores
        df_filtrado[col] = pd.to_numeric(df_filtrado[col], errors='coerce')
    
    # Agrupar por jugador y obtener máximos por columna
    resultado = df_filtrado.groupby('Player Name')[columnas_validas].max().reset_index()
    
    # Renombrar columnas para mayor claridad
    for col in columnas_validas:
        resultado = resultado.rename(columns={col: f"Máximo {col}"})
    
    return resultado

def crear_gauge_carga_aguda(df, jugador_seleccionado=None):
    """
    Crea un gráfico gauge (tacómetro) para mostrar la carga aguda de un jugador.
    
    Parámetros:
    - df: DataFrame con los datos de Catapult
    - jugador_seleccionado: Nombre del jugador (opcional). Si no se proporciona, 
                          se usará el primer jugador disponible.
    
    Retorna:
    - fig: Figura de Plotly con el gráfico gauge
    - datos_procesados: DataFrame con los datos procesados del jugador
    """
    
    # Si no se proporciona un jugador, usar el primero disponible
    if jugador_seleccionado is None:
        if 'Player Name' in df.columns and not df.empty:
            jugador_seleccionado = df['Player Name'].unique()[0]
        else:
            return None, None
    
    # 1. Filtrar datos del jugador
    datos_jugador = df[df['Player Name'] == jugador_seleccionado].copy()
    
    if datos_jugador.empty:
        print(f"No se encontraron datos para el jugador {jugador_seleccionado}")
        return None, None
    
    print(f"Se encontraron {len(datos_jugador)} registros para {jugador_seleccionado}")
    
    # 2. Preparar datos (convertir fechas y columnas numéricas)
    datos_jugador['Date'] = pd.to_datetime(datos_jugador['Date'], errors='coerce')
    
    # Columnas numéricas que necesitamos
    columnas_numericas = [
        'Distance (km)', 
        'Sprint Distance (m)', 
        'Top Speed (m/s)',
        'Accelerations Zone Count: 2 - 3 m/s/s',
        'Accelerations Zone Count: 3 - 4 m/s/s',
        'Accelerations Zone Count: > 4 m/s/s',
        'Deceleration Zone Count: 2 - 3 m/s/s',
        'Deceleration Zone Count: 3 - 4 m/s/s',
        'Deceleration Zone Count: > 4 m/s/s'
    ]
    
    # Convertir columnas a formato numérico
    for col in columnas_numericas:
        if col in datos_jugador.columns:
            # Reemplazar comas por puntos si es string
            if datos_jugador[col].dtype == 'object':
                datos_jugador[col] = datos_jugador[col].str.replace(',', '.', regex=True)
            # Convertir a numérico
            datos_jugador[col] = pd.to_numeric(datos_jugador[col], errors='coerce').fillna(0)
    
    # 3. Calcular carga diaria
    pesos = {
        'Distance (km)': 1.0,
        'Sprint Distance (m)': 0.02,
        'Top Speed (m/s)': 10.0,
        'Accelerations Zone Count: 2 - 3 m/s/s': 1.5,
        'Accelerations Zone Count: 3 - 4 m/s/s': 2.0,
        'Accelerations Zone Count: > 4 m/s/s': 2.5,
        'Deceleration Zone Count: 2 - 3 m/s/s': 1.5,
        'Deceleration Zone Count: 3 - 4 m/s/s': 2.0,
        'Deceleration Zone Count: > 4 m/s/s': 2.5
    }
    
    # Calcular carga diaria aplicando ponderaciones
    datos_jugador['Carga_Diaria'] = 0
    for var, peso in pesos.items():
        if var in datos_jugador.columns:
            datos_jugador['Carga_Diaria'] += datos_jugador[var] * peso
    
    # 4. Obtener últimas 7 sesiones
    datos_ultimos = datos_jugador.sort_values('Date', ascending=False).head(7)
    datos_ultimos = datos_ultimos.sort_values('Date')  # Ordenar cronológicamente para visualización
    
    # 5. Crear gauge chart (tacómetro) con Plotly
    # Calcular promedio de carga de las últimas 7 sesiones para el gauge
    carga_promedio = datos_ultimos['Carga_Diaria'].mean()
    
    # Determinar el valor máximo para la escala del gauge (200% del promedio o un valor fijo)
    max_gauge = max(carga_promedio * 2, datos_ultimos['Carga_Diaria'].max() * 1.2)
    
    # Crear figura con gauge chart
    fig = go.Figure()
    
    # Añadir el indicador gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=datos_ultimos['Carga_Diaria'].iloc[-1],  # Valor más reciente
        title={'text': ""},  # Título simplificado sin el nombre del jugador
        delta={'reference': datos_ultimos['Carga_Diaria'].mean(), 'relative': True, 'valueformat': '.1%'},
        gauge={
            'axis': {'range': [0, max_gauge], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "royalblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_gauge*0.3], 'color': 'lightgreen'},
                {'range': [max_gauge*0.3, max_gauge*0.7], 'color': 'gold'},
                {'range': [max_gauge*0.7, max_gauge], 'color': 'salmon'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': carga_promedio * 1.5  # Umbral en 150% del promedio
            }
        }
    ))

    # Actualizar el layout para mostrar el nombre del jugador en el título principal
    fig.update_layout(
        title=f"Carga Aguda - {jugador_seleccionado}",
        height=500,
        font=dict(size=18),
        margin=dict(l=40, r=40, t=100, b=40),
    )
    
    return fig, datos_jugador

def calcular_carga_cronica(datos_jugador, ventana_cronica=21):
    # Verificar que tenemos datos válidos
    if datos_jugador is None or datos_jugador.empty:
        print("No hay datos para calcular carga crónica")
        return 0, pd.DataFrame()
        
    # Ponderación subjetiva para cada variable (mismos pesos que la carga aguda)
    pesos = {
        'Distance (km)': 1.0,
        'Sprint Distance (m)': 0.02,
        'Top Speed (m/s)': 10.0,
        'Accelerations Zone Count: 2 - 3 m/s/s': 1.5,
        'Accelerations Zone Count: 3 - 4 m/s/s': 2.0,
        'Accelerations Zone Count: > 4 m/s/s': 2.5,
        'Deceleration Zone Count: 2 - 3 m/s/s': 1.5,
        'Deceleration Zone Count: 3 - 4 m/s/s': 2.0,
        'Deceleration Zone Count: > 4 m/s/s': 2.5
    }

    # Asegurarse de que las fechas estén en formato datetime
    datos_jugador['Date'] = pd.to_datetime(datos_jugador['Date'])

    # Ordenar por fecha y aplicar la fórmula de carga diaria si no existe ya
    datos_jugador = datos_jugador.sort_values('Date').copy()
    
    # Verificar si ya existe la columna Carga_Diaria
    if 'Carga_Diaria' not in datos_jugador.columns:
        datos_jugador['Carga_Diaria'] = 0
        for var, peso in pesos.items():
            if var in datos_jugador.columns:
                # Verificar si los datos son numéricos y convertirlos si no lo son
                if datos_jugador[var].dtype == 'object':
                    try:
                        # Reemplazar comas por puntos si es necesario (formato europeo)
                        datos_jugador[var] = datos_jugador[var].str.replace(',', '.', regex=False)
                        datos_jugador[var] = pd.to_numeric(datos_jugador[var], errors='coerce')
                    except Exception as e:
                        print(f"Error al convertir {var}: {e}")
                        # Si hay error, usar 0
                        datos_jugador[var] = 0
                
                # Asegurarse que no hay valores NaN
                datos_temp = datos_jugador[var].fillna(0)
                # Multiplicar por el peso
                try:
                    datos_jugador['Carga_Diaria'] = datos_jugador['Carga_Diaria'] + (datos_temp * peso)
                except Exception as e:
                    print(f"Error al calcular carga para {var}: {e}")
                    # Ignorar esta variable si hay error

    # Calcular la carga crónica como media de los últimos 'ventana_cronica' días
    if 'Carga_Diaria' not in datos_jugador.columns or datos_jugador.empty:
        # Si no existe la columna o no hay datos, devolver 0
        carga_cronica = 0
    elif len(datos_jugador) >= ventana_cronica:
        # Tomar los últimos registros según la ventana
        datos_cronicos = datos_jugador.tail(ventana_cronica)
        
        # Verificar si los datos son numéricos
        if datos_cronicos['Carga_Diaria'].dtype == 'object':
            try:
                datos_cronicos['Carga_Diaria'] = pd.to_numeric(datos_cronicos['Carga_Diaria'], errors='coerce')
            except Exception as e:
                print(f"Error al convertir Carga_Diaria a numérico: {e}")
        
        # Calcular la media de la carga diaria, manejando valores nulos
        carga_cronica = datos_cronicos['Carga_Diaria'].fillna(0).mean()
    else:
        # Si no hay suficientes datos, usar todos los disponibles
        # Asegurarse de que sean numéricos
        if datos_jugador['Carga_Diaria'].dtype == 'object':
            try:
                datos_jugador['Carga_Diaria'] = pd.to_numeric(datos_jugador['Carga_Diaria'], errors='coerce')
            except Exception as e:
                print(f"Error al convertir Carga_Diaria a numérico: {e}")
        
        carga_cronica = datos_jugador['Carga_Diaria'].fillna(0).mean()
    
    # Verificar que la carga crónica sea un número válido
    if pd.isna(carga_cronica) or not isinstance(carga_cronica, (int, float)):
        print("La carga crónica calculada no es un número válido, retornando 0")
        carga_cronica = 0
    
    return carga_cronica, datos_jugador

# Función para cargar datos
@st.cache_data
def load_data():
    sheet_id = "1kajUuZwL9l1suipRNy7t2g_SGJUcDVh6MeTu66yK-Z4"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    
    try:
        df = pd.read_csv(url)
        # Verificar si hay datos
        if df is None or df.empty:
            st.warning("No se encontraron datos en la fuente.")
            return None
            
        # Convertir columnas numéricas a formato numérico
        # Lista de columnas que deben ser numéricas
        columnas_numericas = [
            'Distance (km)', 'Sprint Distance (m)', 'Top Speed (m/s)',
            'Accelerations Zone Count: 2 - 3 m/s/s', 'Accelerations Zone Count: 3 - 4 m/s/s',
            'Accelerations Zone Count: > 4 m/s/s', 'Deceleration Zone Count: 2 - 3 m/s/s',
            'Deceleration Zone Count: 3 - 4 m/s/s', 'Deceleration Zone Count: > 4 m/s/s'
        ]
        
        for col in columnas_numericas:
            if col in df.columns:
                # Convertir valores de formato europeo (comas) a punto decimal
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.replace(',', '.', regex=False)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        print(f"Error al convertir columna {col}: {e}")
                        
        # Asegurarse de que la fecha esté en formato correcto
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Función para obtener la lista de jugadores
def obtener_jugadores(df):
    if df is None or df.empty:
        return []
    
    if 'Player Name' in df.columns:
        jugadores = df['Player Name'].unique().tolist()
    elif 'Given Name' in df.columns and 'Family Name' in df.columns:
        df['Player Name'] = df['Given Name'] + ' ' + df['Family Name']
        jugadores = df['Player Name'].unique().tolist()
    else:
        st.error("No se encontraron columnas con nombres de jugadores")
        return []
    
    return sorted(jugadores)

# Función para procesar datos de un jugador específico
def procesar_datos_jugador(df, nombre_jugador):
    if df is None or df.empty:
        st.warning("No hay datos disponibles para analizar.")
        return None
    
    # Filtrar datos del jugador seleccionado
    if 'Player Name' in df.columns:
        datos_jugador = df[df['Player Name'] == nombre_jugador].copy()
    else:
        st.error("No se encontró la columna 'Player Name'")
        return None
    
    if datos_jugador.empty:
        st.warning(f"No se encontraron registros para {nombre_jugador}")
        return None
    
    # Convertir columnas necesarias
    try:
        # Convertir fecha a datetime si no lo está
        if not pd.api.types.is_datetime64_dtype(datos_jugador['Date']):
            datos_jugador['Date'] = pd.to_datetime(datos_jugador['Date'], errors='coerce')
          # Convertir distancia a numérica
        if 'Distance (km)' in datos_jugador.columns and not datos_jugador.empty:
            # Verificar primero si hay datos no nulos
            if not datos_jugador['Distance (km)'].dropna().empty:
                # Verificar tipo del primer valor no nulo
                first_valid = datos_jugador['Distance (km)'].dropna().iloc[0]
                if isinstance(first_valid, str):
                    try:
                        datos_jugador['Distance (km)'] = datos_jugador['Distance (km)'].str.replace(',', '.', regex=False)
                        datos_jugador['Distance (km)'] = pd.to_numeric(datos_jugador['Distance (km)'], errors='coerce')
                    except Exception as e:
                        st.warning(f"Error al convertir Distance (km): {e}")
            
            # Si no es numérico, intentar convertir de todas formas
            if not pd.api.types.is_numeric_dtype(datos_jugador['Distance (km)']):
                try:
                    datos_jugador['Distance (km)'] = pd.to_numeric(datos_jugador['Distance (km)'], errors='coerce')
                except Exception as e:
                    st.warning(f"Error al convertir Distance (km) a numérico: {e}")
          # Procesar métricas de velocidad alta
        columnas_velocidad = [
            'Sprint Distance (m)',
        ]
        
        for col in columnas_velocidad:
            if col in datos_jugador.columns and not datos_jugador.empty:
                # Verificar primero si hay datos no nulos
                if not datos_jugador[col].dropna().empty:
                    # Verificar tipo del primer valor no nulo
                    try:
                        first_valid = datos_jugador[col].dropna().iloc[0]
                        if isinstance(first_valid, str):
                            datos_jugador[col] = datos_jugador[col].str.replace(',', '.', regex=False)
                            datos_jugador[col] = pd.to_numeric(datos_jugador[col], errors='coerce')
                    except Exception as e:
                        st.warning(f"Error al procesar {col}: {e}")
                
                # Si no es numérico, intentar convertir de todas formas
                if not pd.api.types.is_numeric_dtype(datos_jugador[col]):
                    try:
                        datos_jugador[col] = pd.to_numeric(datos_jugador[col], errors='coerce')
                    except Exception as e:
                        st.warning(f"Error al convertir {col} a numérico: {e}")
        
        # Calcular suma de alta velocidad solo si las columnas existen
        if all(col in datos_jugador.columns for col in columnas_velocidad):
            try:
                datos_jugador['Alta Velocidad (km)'] = datos_jugador[columnas_velocidad].fillna(0).sum(axis=1)
            except Exception as e:
                st.warning(f"Error al calcular Alta Velocidad: {e}")
                datos_jugador['Alta Velocidad (km)'] = 0
          # Procesar métricas de aceleración
        columnas_aceleraciones = [
            'Accelerations Zone Count: 2 - 3 m/s/s', 
            'Accelerations Zone Count: 3 - 4 m/s/s', 
            'Accelerations Zone Count: > 4 m/s/s'
        ]
        
        columnas_desaceleraciones = [
            'Deceleration Zone Count: 1 - 2 m/s/s',
            'Deceleration Zone Count: 2 - 3 m/s/s',
            'Deceleration Zone Count: 3 - 4 m/s/s',
            'Deceleration Zone Count: > 4 m/s/s'
        ]
        
        # Procesamiento seguro de las columnas de aceleración/desaceleración
        for col in columnas_aceleraciones + columnas_desaceleraciones:
            if col in datos_jugador.columns and not datos_jugador.empty:
                # Verificar si hay datos no nulos
                if not datos_jugador[col].dropna().empty:
                    try:
                        first_valid = datos_jugador[col].dropna().iloc[0]
                        if isinstance(first_valid, str):
                            datos_jugador[col] = datos_jugador[col].str.replace(',', '.', regex=False)
                            datos_jugador[col] = pd.to_numeric(datos_jugador[col], errors='coerce')
                    except Exception as e:
                        st.warning(f"Error al procesar {col}: {e}")
                
                # Si no es numérico, intentar convertir de todas formas
                if not pd.api.types.is_numeric_dtype(datos_jugador[col]):
                    try:
                        datos_jugador[col] = pd.to_numeric(datos_jugador[col], errors='coerce')
                    except Exception as e:
                        st.warning(f"Error al convertir {col} a numérico: {e}")
        
        # Calcular sumas totales solo si todas las columnas existen
        columnas_aceleraciones_existentes = [col for col in columnas_aceleraciones if col in datos_jugador.columns]
        if columnas_aceleraciones_existentes:
            try:
                datos_jugador['Total Aceleraciones'] = datos_jugador[columnas_aceleraciones_existentes].fillna(0).sum(axis=1)
            except Exception as e:
                st.warning(f"Error al calcular Total Aceleraciones: {e}")
                datos_jugador['Total Aceleraciones'] = 0
        
        columnas_desaceleraciones_existentes = [col for col in columnas_desaceleraciones if col in datos_jugador.columns]
        if columnas_desaceleraciones_existentes:
            try:
                datos_jugador['Total Desaceleraciones'] = datos_jugador[columnas_desaceleraciones_existentes].fillna(0).sum(axis=1)
            except Exception as e:
                st.warning(f"Error al calcular Total Desaceleraciones: {e}")
                datos_jugador['Total Desaceleraciones'] = 0
        
        # Ordenar por fecha
        datos_jugador = datos_jugador.sort_values('Date', ascending=True)
        
        return datos_jugador
    
    except Exception as e:
        st.error(f"Error al procesar datos del jugador: {e}")
        return None

# Función para graficar la evolución de distancia por sesión para un jugador
def graficar_distancia_jugador(datos_jugador, n_sesiones=10):
    """
    Crea un gráfico de barras para visualizar la evolución de la distancia recorrida
    por un jugador en sus últimas sesiones.
    
    Parámetros:
    - datos_jugador: DataFrame con los datos del jugador
    - n_sesiones: Número de últimas sesiones a mostrar
    
    Retorna:
    - fig: Figura de Plotly para visualización
    """
    # Verificar si hay datos válidos
    if datos_jugador is None or datos_jugador.empty:
        st.warning("No hay datos para visualizar")
        return None
    
    # Verificar que las columnas necesarias estén presentes
    columnas_requeridas = ['Date', 'Distance (km)', 'Session Title']
    columnas_faltantes = [col for col in columnas_requeridas if col not in datos_jugador.columns]
    
    if columnas_faltantes:
        st.warning(f"Faltan las siguientes columnas: {', '.join(columnas_faltantes)}")
        return None
    
    try:
        # Verificar tipo de datos de fecha
        if not pd.api.types.is_datetime64_dtype(datos_jugador['Date']):
            datos_jugador['Date'] = pd.to_datetime(datos_jugador['Date'], errors='coerce')
        
        # Eliminar filas donde Date es NaT (fecha no válida)
        datos_jugador = datos_jugador.dropna(subset=['Date'])
        
        # Verificar si aún quedan datos después de eliminar fechas no válidas
        if datos_jugador.empty:
            st.warning("No hay fechas válidas en los datos")
            return None
        
        # Ordenar datos por fecha de más reciente a más antigua
        datos_jugador_ordenado = datos_jugador.sort_values('Date', ascending=False)
        
        # Tomar las últimas n sesiones (ahora serán las más recientes)
        if len(datos_jugador_ordenado) > n_sesiones:
            datos_plot = datos_jugador_ordenado.head(n_sesiones)
        else:
            datos_plot = datos_jugador_ordenado.copy()
        
        # Reordenar para la visualización (de más antigua a más reciente para el eje X)
        datos_plot = datos_plot.sort_values('Date')
        
        # Verificar si 'Distance (km)' es numérico, si no, convertirlo
        if not pd.api.types.is_numeric_dtype(datos_plot['Distance (km)']):
            datos_plot['Distance (km)'] = pd.to_numeric(datos_plot['Distance (km)'], errors='coerce')
            
        # Rellenar valores NaN en Distance con 0
        datos_plot['Distance (km)'] = datos_plot['Distance (km)'].fillna(0)
        
        # Asegurarse de que Session Title tenga valores (no nulos)
        if 'Session Title' in datos_plot.columns:
            datos_plot['Session Title'] = datos_plot['Session Title'].fillna('Sin título')
        else:
            datos_plot['Session Title'] = 'Sin título'
        
        # Crear etiquetas personalizadas con fecha y título de sesión
        etiquetas_eje_x = [f"{fecha.strftime('%d/%m/%Y')}<br><b>{sesion}</b>" 
                          for fecha, sesion in zip(datos_plot['Date'], datos_plot['Session Title'])]
        
        # Crear figura con Plotly
        fig = go.Figure()
        
        # Añadir barras para distancia con texto personalizado
        fig.add_trace(go.Bar(
            x=datos_plot['Date'].dt.date,
            y=datos_plot['Distance (km)'],
            name='Distancia',
            marker_color='rgb(70, 70, 70)',
            text=[f"{dist:.2f} km<br><b>{session}</b>" for dist, session in zip(datos_plot['Distance (km)'], datos_plot['Session Title'])],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br><span style="font-size:16px;"><b>%{customdata}</b></span><br>Distancia: %{y:.2f} km<extra></extra>',
            customdata=datos_plot['Session Title']
        ))
        
        # Agregar línea de tendencia
        fig.add_trace(go.Scatter(
            x=datos_plot['Date'].dt.date,
            y=datos_plot['Distance (km)'],
            name='Tendencia',
            mode='lines',
            line=dict(color='rgb(30, 30, 30)', width=3, shape='spline'),
            hovertemplate='<b>%{x}</b><br>Tendencia: %{y:.2f} km<extra></extra>'
        ))
        
        # Actualizar diseño con etiquetas personalizadas
        fig.update_layout(
            title={
                'text': 'Evolución de Distancia por Sesión',
                'font': {'size': 24},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title='Fecha',
                tickangle=-30,
                tickfont=dict(size=14),
                title_font=dict(size=16),
                # Usar etiquetas personalizadas
                ticktext=etiquetas_eje_x,
                tickvals=datos_plot['Date'].dt.date
            ),
            yaxis=dict(
                title='Distancia (km)',
                gridcolor='rgb(220, 220, 220)'
            ),
            plot_bgcolor='rgb(250, 250, 250)',
            paper_bgcolor='rgb(250, 250, 250)',
            font=dict(color='rgb(20, 20, 20)'),
            bargap=0.2,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=600,  # Aumentar la altura para dar más espacio a las etiquetas
            margin=dict(b=150)  # Aumentar el margen inferior para las etiquetas
        )
        
        return fig
    except Exception as e:
        st.error(f"Error al generar gráfico de distancia: {e}")
        return None

# Función para graficar la evolución de velocidad alta por sesión para un jugador
def graficar_velocidad_jugador(datos_jugador, n_sesiones=10):
    """
    Crea un gráfico de barras para visualizar la evolución de sprint distance
    por un jugador en sus últimas sesiones.
    
    Parámetros:
    - datos_jugador: DataFrame con los datos del jugador
    - n_sesiones: Número de últimas sesiones a mostrar
    
    Retorna:
    - fig: Figura de Plotly para visualización
    """
    # Verificar si hay datos válidos
    if datos_jugador is None or datos_jugador.empty:
        st.warning("No hay datos para visualizar")
        return None
    
    # Verificar que las columnas necesarias estén presentes
    columnas_requeridas = ['Date', 'Sprint Distance (m)', 'Session Title']
    columnas_faltantes = [col for col in columnas_requeridas if col not in datos_jugador.columns]
    
    if columnas_faltantes:
        st.warning(f"Faltan las siguientes columnas para el gráfico de velocidad: {', '.join(columnas_faltantes)}")
        return None
    
    try:
        # Verificar tipo de datos de fecha
        if not pd.api.types.is_datetime64_dtype(datos_jugador['Date']):
            datos_jugador['Date'] = pd.to_datetime(datos_jugador['Date'], errors='coerce')
        
        # Eliminar filas donde Date es NaT (fecha no válida)
        datos_jugador = datos_jugador.dropna(subset=['Date'])
        
        # Verificar si aún quedan datos después de eliminar fechas no válidas
        if datos_jugador.empty:
            st.warning("No hay fechas válidas en los datos")
            return None
        
        # Ordenar datos por fecha de más reciente a más antigua
        datos_jugador_ordenado = datos_jugador.sort_values('Date', ascending=False)
        
        # Tomar las últimas n sesiones (ahora serán las más recientes)
        if len(datos_jugador_ordenado) > n_sesiones:
            datos_plot = datos_jugador_ordenado.head(n_sesiones)
        else:
            datos_plot = datos_jugador_ordenado.copy()
        
        # Reordenar para la visualización (de más antigua a más reciente para el eje X)
        datos_plot = datos_plot.sort_values('Date')
        
        # Verificar si 'Sprint Distance (m)' es numérico, si no, convertirlo
        if not pd.api.types.is_numeric_dtype(datos_plot['Sprint Distance (m)']):
            datos_plot['Sprint Distance (m)'] = pd.to_numeric(datos_plot['Sprint Distance (m)'], errors='coerce')
            
        # Rellenar valores NaN en Sprint Distance con 0
        datos_plot['Sprint Distance (m)'] = datos_plot['Sprint Distance (m)'].fillna(0)
        
        # Asegurarse de que Session Title tenga valores (no nulos)
        if 'Session Title' in datos_plot.columns:
            datos_plot['Session Title'] = datos_plot['Session Title'].fillna('Sin título')
        else:
            datos_plot['Session Title'] = 'Sin título'
        
        # Crear etiquetas personalizadas con fecha y título de sesión
        etiquetas_eje_x = [f"{fecha.strftime('%d/%m/%Y')}<br><b>{sesion}</b>" 
                          for fecha, sesion in zip(datos_plot['Date'], datos_plot['Session Title'])]
        
        # Crear figura con Plotly
        fig = go.Figure()
        
        # Añadir barras para sprint distance con texto personalizado
        fig.add_trace(go.Bar(
            x=datos_plot['Date'].dt.date,
            y=datos_plot['Sprint Distance (m)'],
            name='Sprint Distance',
            marker_color='rgb(65, 105, 225)',  # Azul royal
            text=[f"{sprint:.2f} m<br><b>{session}</b>" for sprint, session in zip(datos_plot['Sprint Distance (m)'], datos_plot['Session Title'])],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br><span style="font-size:16px;"><b>%{customdata}</b></span><br>Sprint Distance: %{y:.2f} m<extra></extra>',
            customdata=datos_plot['Session Title']
        ))
        
        # Agregar línea de tendencia
        fig.add_trace(go.Scatter(
            x=datos_plot['Date'].dt.date,
            y=datos_plot['Sprint Distance (m)'],
            name='Tendencia',
            mode='lines',
            line=dict(color='rgb(30, 30, 30)', width=3, shape='spline'),
            hovertemplate='<b>%{x}</b><br>Tendencia: %{y:.2f} m<extra></extra>'
        ))
        
        # Actualizar diseño con etiquetas personalizadas
        fig.update_layout(
            title={
                'text': 'Evolución de Sprint Distance por Sesión',
                'font': {'size': 24},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title='Fecha',
                tickangle=-30,
                tickfont=dict(size=14),
                title_font=dict(size=16),
                # Usar etiquetas personalizadas
                ticktext=etiquetas_eje_x,
                tickvals=datos_plot['Date'].dt.date
            ),
            yaxis=dict(
                title='Sprint Distance (m)',
                gridcolor='rgb(220, 220, 220)'
            ),
            plot_bgcolor='rgb(250, 250, 250)',
            paper_bgcolor='rgb(250, 250, 250)',
            font=dict(color='rgb(20, 20, 20)'),
            bargap=0.2,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=600,  # Aumentar la altura para dar más espacio a las etiquetas
            margin=dict(b=150)  # Aumentar el margen inferior para las etiquetas
        )
        
        return fig
    except Exception as e:
        st.error(f"Error al generar gráfico de velocidad: {e}")
        return None

# Función para graficar la evolución de aceleraciones por sesión para un jugador
def graficar_aceleraciones_jugador(datos_jugador, n_sesiones=10):
    if datos_jugador is None or datos_jugador.empty:
        st.warning("No hay datos de aceleraciones para visualizar")
        return None
    
    if 'Total Aceleraciones' not in datos_jugador.columns or 'Total Desaceleraciones' not in datos_jugador.columns:
        st.warning("No se encontraron datos de aceleraciones o desaceleraciones")
        return None
    
    # Verificar que 'Date' y 'Session Title' estén presentes
    columnas_requeridas = ['Date', 'Session Title']
    columnas_faltantes = [col for col in columnas_requeridas if col not in datos_jugador.columns]
    
    if columnas_faltantes:
        st.warning(f"Faltan las siguientes columnas: {', '.join(columnas_faltantes)}")
        return None
    
    # Ordenar datos por fecha de más reciente a más antigua
    datos_jugador_ordenado = datos_jugador.sort_values('Date', ascending=False)
    
    # Tomar las últimas n sesiones (ahora serán las más recientes)
    if len(datos_jugador_ordenado) > n_sesiones:
        datos_plot = datos_jugador_ordenado.head(n_sesiones)
    else:
        datos_plot = datos_jugador_ordenado.copy()
    
    # Reordenar para la visualización (de más antigua a más reciente para el eje X)
    datos_plot = datos_plot.sort_values('Date')
    
    # Asegurarse de que Session Title tenga valores (no nulos)
    if 'Session Title' in datos_plot.columns:
        datos_plot['Session Title'] = datos_plot['Session Title'].fillna('Sin título')
    else:
        datos_plot['Session Title'] = 'Sin título'
    
    # Crear etiquetas personalizadas con fecha y título de sesión
    etiquetas_eje_x = [f"{fecha.strftime('%d/%m/%Y')}<br><b>{sesion}</b>" 
                      for fecha, sesion in zip(datos_plot['Date'], datos_plot['Session Title'])]
    
    # Crear figura con Plotly
    fig = go.Figure()
    
    # Añadir barras para aceleraciones con texto personalizado
    fig.add_trace(go.Bar(
        x=datos_plot['Date'].dt.date,
        y=datos_plot['Total Aceleraciones'],
        name='Aceleraciones',
        marker_color='rgb(65, 105, 225)',  # Azul royal
        text=[f"{acel}<br><b>{session}</b>" for acel, session in 
              zip(datos_plot['Total Aceleraciones'].round(0).astype(int), 
                  datos_plot['Session Title'])],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br><span style="font-size:16px;"><b>%{customdata}</b></span><br>Aceleraciones: %{y}<extra></extra>',
        customdata=datos_plot['Session Title']
    ))
    
    # Añadir barras para desaceleraciones con texto personalizado
    fig.add_trace(go.Bar(
        x=datos_plot['Date'].dt.date,
        y=datos_plot['Total Desaceleraciones'],
        name='Desaceleraciones',
        marker_color='rgb(220, 20, 60)',  # Rojo carmesí
        text=[f"{desacel}<br><b>{session}</b>" for desacel, session in 
              zip(datos_plot['Total Desaceleraciones'].round(0).astype(int), 
                  datos_plot['Session Title'])],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br><span style="font-size:16px;"><b>%{customdata}</b></span><br>Desaceleraciones: %{y}<extra></extra>',
        customdata=datos_plot['Session Title']
    ))
    
    # Actualizar diseño con etiquetas personalizadas
    fig.update_layout(
        title={
            'text': 'Evolución de Aceleraciones y Desaceleraciones por Sesión',
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Fecha',
            tickangle=-30,
            tickfont=dict(size=14),
            title_font=dict(size=16),
            # Usar etiquetas personalizadas
            ticktext=etiquetas_eje_x,
            tickvals=datos_plot['Date'].dt.date
        ),
        yaxis=dict(
            title='Cantidad',
            gridcolor='rgb(220, 220, 220)'
        ),
        plot_bgcolor='rgb(250, 250, 250)',
        paper_bgcolor='rgb(250, 250, 250)',
        font=dict(color='rgb(20, 20, 20)'),
        barmode='group',  # Barras agrupadas
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=600,  # Aumentar la altura para dar más espacio a las etiquetas
        margin=dict(b=150)  # Aumentar el margen inferior para las etiquetas
    )
    
    return fig

# Función para crear una tabla de resumen de estadísticas por tipo de sesión
def crear_tabla_resumen(datos_jugador):
    if datos_jugador is None or datos_jugador.empty:
        st.warning("No hay datos suficientes para mostrar información")
        return None
    
    # Verificar columnas necesarias
    columnas_requeridas = [
        'Session Title', 
        'Distance (km)',
        'Top Speed (m/s)',
        'Sprint Distance (m)',
        'Accelerations Zone Count: 2 - 3 m/s/s', 
        'Accelerations Zone Count: 3 - 4 m/s/s', 
        'Accelerations Zone Count: > 4 m/s/s',
        'Deceleration Zone Count: 1 - 2 m/s/s',
        'Deceleration Zone Count: 2 - 3 m/s/s',
        'Deceleration Zone Count: 3 - 4 m/s/s',
        'Deceleration Zone Count: > 4 m/s/s'
    ]
    
    # Verificar qué columnas están disponibles
    columnas_disponibles = [col for col in columnas_requeridas if col in datos_jugador.columns]
    
    if len(columnas_disponibles) <= 1:  # Solo tiene 'Session Title' o ni siquiera eso
        st.warning("No hay suficientes columnas para generar la tabla de resumen")
        return None
    
    # Crear una copia del DataFrame con solo las columnas que nos interesan
    resumen = datos_jugador[columnas_disponibles].copy()
    
    # Renombrar columnas para mejor visualización
    renombres = {
        'Session Title': 'Tipo de Sesión',
        'Distance (km)': 'Distancia (km)',
        'Top Speed (m/s)': 'Velocidad Máxima (m/s)',
        'Sprint Distance (m)': 'Distancia Sprint (m)',
        'Accelerations Zone Count: 2 - 3 m/s/s': 'Aceleraciones 2-3 m/s/s',
        'Accelerations Zone Count: 3 - 4 m/s/s': 'Aceleraciones 3-4 m/s/s',
        'Accelerations Zone Count: > 4 m/s/s': 'Aceleraciones >4 m/s/s',
        'Deceleration Zone Count: 1 - 2 m/s/s': 'Desaceleraciones 1-2 m/s/s',
        'Deceleration Zone Count: 2 - 3 m/s/s': 'Desaceleraciones 2-3 m/s/s',
        'Deceleration Zone Count: 3 - 4 m/s/s': 'Desaceleraciones 3-4 m/s/s',
        'Deceleration Zone Count: > 4 m/s/s': 'Desaceleraciones >4 m/s/s'
    }
    
    resumen = resumen.rename(columns={k: v for k, v in renombres.items() if k in resumen.columns})
    
    # Redondear valores numéricos para mejor visualización
    for col in resumen.columns:
        if col != 'Tipo de Sesión':
            if ('Aceleraciones' in col or 'Desaceleraciones' in col):
                # Para aceleraciones y desaceleraciones, mostrar como enteros
                if pd.api.types.is_numeric_dtype(resumen[col]):
                    resumen[col] = resumen[col].round(0).astype(int)
            elif 'Distancia Sprint' in col:
                # Para distancias sprint, 1 decimal
                if pd.api.types.is_numeric_dtype(resumen[col]):
                    resumen[col] = resumen[col].round(1)
            elif pd.api.types.is_numeric_dtype(resumen[col]):
                # Para otras métricas numéricas, 2 decimales
                resumen[col] = resumen[col].round(2)
    
    return resumen

# Función para mostrar métricas del jugador en las últimas sesiones
def mostrar_metricas_recientes(datos_jugador, dias=30):
    if datos_jugador is None or datos_jugador.empty:
        st.warning("No hay datos para mostrar métricas recientes")
        return
    
    # Calcular fecha límite (últimos X días)
    fecha_limite = datetime.now() - timedelta(days=dias)
    
    # Filtrar datos recientes
    datos_recientes = datos_jugador[datos_jugador['Date'] >= fecha_limite]
    
    if datos_recientes.empty:
        st.warning(f"No hay datos en los últimos {dias} días")
        return
    
    # Crear métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        distancia_total = datos_recientes['Distance (km)'].sum().round(2)
        st.metric(label="Distancia Total (km)", value=distancia_total)
    
    with col2:
        velocidad_alta_total = datos_recientes['Alta Velocidad (km)'].sum().round(2)
        st.metric(label="Velocidad Alta Total (km)", value=velocidad_alta_total)
    
    with col3:
        aceleraciones_promedio = datos_recientes['Total Aceleraciones'].mean().round(0).astype(int)
        st.metric(label="Aceleraciones Promedio", value=aceleraciones_promedio)
    
    with col4:
        desaceleraciones_promedio = datos_recientes['Total Desaceleraciones'].mean().round(0).astype(int)
        st.metric(label="Desaceleraciones Promedio", value=desaceleraciones_promedio)

# Función para obtener estadísticas de velocidad por partido
def obtener_estadisticas_velocidad(df_partidos):
    """
    Calcula estadísticas de velocidad alta por partido.
    
    Parámetros:
    - df_partidos: DataFrame con datos de partidos
    
    Retorna:
    - DataFrame con estadísticas de velocidad alta por partido
    """
    if df_partidos.empty:
        return pd.DataFrame()
    
    # Verificar que las columnas necesarias existen
    columnas_necesarias = ['Date', 'Session Title', 'Player Name', 'Alta Velocidad (km)']
    if not all(col in df_partidos.columns for col in columnas_necesarias):
        print("Faltan columnas necesarias para calcular estadísticas de velocidad")
        return pd.DataFrame()
    
    # Agrupar por fecha y partido
    resultados = []
    
    for fecha, grupo_fecha in df_partidos.groupby('Date'):
        for partido, grupo_partido in grupo_fecha.groupby('Session Title'):
            # Calcular estadísticas
            velocidad_alta_total = grupo_partido['Alta Velocidad (km)'].sum()
            velocidad_alta_media = grupo_partido['Alta Velocidad (km)'].mean()
            velocidad_alta_max = grupo_partido['Alta Velocidad (km)'].max()
            num_jugadores = grupo_partido['Player Name'].nunique()
            
            resultados.append({
                'Fecha': fecha,
                'Partido': partido,
                'Velocidad Alta Total (km)': velocidad_alta_total,
                'Velocidad Alta Media (km)': velocidad_alta_media,
                'Velocidad Alta Máxima (km)': velocidad_alta_max,
                'N° Jugadores': num_jugadores
            })
    
    # Convertir a DataFrame y ordenar por fecha
    resultados_df = pd.DataFrame(resultados)
    if not resultados_df.empty:
        resultados_df = resultados_df.sort_values('Fecha')
    
    return resultados_df

# Función para mostrar el panel de carga con tres columnas
def mostrar_panel_carga(datos_jugador, jugador_seleccionado):
    """
    Muestra un panel con tres columnas para visualizar valores máximos, carga aguda y carga crónica.
    
    Parámetros:
    - datos_jugador: DataFrame con los datos del jugador seleccionado
    - jugador_seleccionado: Nombre del jugador seleccionado
    """
    st.markdown("## 📊 Panel de Análisis de Carga")
    
    # Crear las tres columnas
    col1, col2, col3 = st.columns(3)
    
    # Columna 1: Valores máximos históricos
    with col1:
        st.markdown("### 🏆 Valores Máximos")
        if datos_jugador is not None and not datos_jugador.empty:
            # Columnas relevantes para mostrar máximos
            columnas_maximos = [
                'Distance (km)', 
                'Sprint Distance (m)',
                'Top Speed (m/s)',
                'Accelerations Zone Count: > 4 m/s/s',
                'Deceleration Zone Count: > 4 m/s/s'
            ]
            
            # Obtener máximos para este jugador
            df_maximos = obtener_valores_maximos_por_jugador(
                datos_jugador, columnas_maximos, jugador_seleccionado)
            
            if not df_maximos.empty:
                # Crear un recuadro para cada valor máximo
                for col in df_maximos.columns:
                    if col != 'Player Name':
                        nombre_metrica = col.replace('Máximo ', '')
                        valor = df_maximos[col].iloc[0]
                        
                        # Unidades según el tipo de métrica
                        unidad = ""
                        if "Distance (km)" in col:
                            unidad = "km"
                            valor = round(valor, 2)
                        elif "Distance (m)" in col:
                            unidad = "m"
                            valor = round(valor, 1)
                        elif "Speed" in col:
                            unidad = "m/s"
                            valor = round(valor, 2)
                        else:
                            valor = int(valor)
                        
                        # Crear tarjeta para el valor máximo
                        st.markdown(f"""
                        <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                            <p style="font-size: 16px; margin-bottom: 5px;">{nombre_metrica}</p>
                            <p style="font-size: 24px; font-weight: bold; margin: 0;">{valor} {unidad}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No se pudieron calcular los valores máximos")
        else:
            st.warning("No hay datos disponibles")
    
    # Columna 2: Carga aguda (últimos 7 días)
    with col2:
        st.markdown("### 🔥 Carga Aguda (7 días)")
        if datos_jugador is not None and not datos_jugador.empty:
            # Usar la función existente crear_gauge_carga_aguda
            fig_gauge, _ = crear_gauge_carga_aguda(datos_jugador, jugador_seleccionado)
            if fig_gauge:
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Extraer datos de las últimas 7 sesiones
                datos_ultimos = datos_jugador.sort_values('Date', ascending=False).head(7)
                datos_ultimos = datos_ultimos.sort_values('Date')
                
                # Calcular y mostrar promedio de carga aguda
                if 'Carga_Diaria' in datos_ultimos.columns:
                    carga_promedio = datos_ultimos['Carga_Diaria'].mean()
                    st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                        <p style="font-size: 16px; margin-bottom: 5px;">Promedio Carga Diaria (7 días)</p>
                        <p style="font-size: 24px; font-weight: bold; margin: 0;">{carga_promedio:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No se pudo generar el gráfico de carga aguda")
        else:
            st.warning("No hay datos disponibles")
    
    with col3:
        st.markdown("### 📈 Carga Crónica (21 días)")
        if datos_jugador is not None and not datos_jugador.empty:
            # Usar la función existente calcular_carga_cronica
            carga_cronica, datos_con_carga = calcular_carga_cronica(datos_jugador)
            
            if carga_cronica:
                # Crear un gauge para carga crónica
                max_gauge = carga_cronica * 2
                
                # Crear figura con gauge chart para carga crónica
                fig = go.Figure()
                
                # Añadir el indicador gauge
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",  # Cambiado para coincidir con carga aguda
                    value=carga_cronica,
                    title={'text': ""},  # Título simplificado como en carga aguda
                    delta={'reference': carga_cronica * 0.9, 'relative': True, 'valueformat': '.1%'},  # Añadido para coincidir con carga aguda
                    gauge={
                        'axis': {'range': [0, max_gauge], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "royalblue"},  # Cambiado de "darkgreen" a "royalblue" para coincidir
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, max_gauge*0.3], 'color': 'lightgreen'},
                            {'range': [max_gauge*0.3, max_gauge*0.7], 'color': 'gold'},
                            {'range': [max_gauge*0.7, max_gauge], 'color': 'salmon'}
                        ],
                        'threshold': {  # Añadido para coincidir con carga aguda
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': carga_cronica * 1.5  # Umbral en 150% del valor crónico
                        }
                    }
                ))
                
                # Actualizar el layout para coincidir con el de carga aguda
                fig.update_layout(
                    title=f"Carga Crónica - {jugador_seleccionado}",
                    height=500,  # Cambiado a 500 para coincidir con carga aguda
                    font=dict(size=18),  # Cambiado a 18 para coincidir
                    margin=dict(l=40, r=40, t=100, b=40),  # Actualizado t a 100 para coincidir
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Se eliminó toda la sección de la relación aguda:crónica
                
            else:
                st.warning("No se pudo calcular la carga crónica")
        else:
            st.warning("No hay datos disponibles")

# Función para visualizar estadísticas de velocidad por partido
def visualizar_estadisticas_velocidad(df):
    """
    Visualiza las estadísticas de velocidad alta en un gráfico de barras.
    
    Parámetros:
    - df: DataFrame con los datos de Catapult
    
    Retorna:
    - Figura de Plotly con el gráfico de velocidad alta por partido
    """
    try:
        # Primero filtrar solo partidos
        partidos_df = df[df['Session Title'].str.contains('Partido', case=False, na=False)].copy()
        
        if partidos_df.empty:
            st.warning("No se encontraron datos de partidos para visualizar.")
            return None
        
        # Convertir fecha a datetime si no lo está
        if not pd.api.types.is_datetime64_dtype(partidos_df['Date']):
            partidos_df['Date'] = pd.to_datetime(partidos_df['Date'], errors='coerce')
        
        # Obtener estadísticas de velocidad
        velocidad_stats = obtener_estadisticas_velocidad(partidos_df)
        
        if velocidad_stats.empty:
            st.warning("No se pudieron calcular estadísticas de velocidad para los partidos.")
            return None
        
        # Crear identificador único para cada partido (fecha + nombre)
        velocidad_stats['ID_Partido'] = velocidad_stats['Fecha'].dt.strftime('%d/%m/%Y') + ' - ' + velocidad_stats['Partido']
        
        # Limitar a los últimos 10 partidos para mejor visualización
        if len(velocidad_stats) > 10:
            velocidad_stats = velocidad_stats.tail(10)
        
        # Crear gráfico de barras con Plotly
        fig = go.Figure()
        
        # Añadir barras para la velocidad alta media
        fig.add_trace(go.Bar(
            x=velocidad_stats['ID_Partido'],
            y=velocidad_stats['Velocidad Alta Media (km)'],
            name='Velocidad Alta Media',
            marker_color='rgb(70, 70, 70)',
            text=velocidad_stats['Velocidad Alta Media (km)'].round(2),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Velocidad Alta Media: %{y:.2f} km<br>N° Jugadores: %{customdata[0]}<br>Total: %{customdata[1]:.2f} km<extra></extra>',
            customdata=velocidad_stats[['N° Jugadores', 'Velocidad Alta Total (km)']]
        ))
        
        # Añadir línea de tendencia
        fig.add_trace(go.Scatter(
            x=velocidad_stats['ID_Partido'],
            y=velocidad_stats['Velocidad Alta Media (km)'],
            name='Tendencia',
            mode='lines+markers',
            line=dict(color='rgb(30, 30, 30)', width=2),
            marker=dict(color='rgb(30, 30, 30)', size=8),
            hoverinfo='skip'
        ))
        
        # Configurar diseño
        fig.update_layout(
            title={
                'text': 'Velocidad Alta Media por Partido',
                'font': {'size': 24},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title='Partido',
                tickangle=-45,
                tickfont=dict(size=12),
                title_font=dict(size=16)
            ),
            yaxis=dict(
                title='Velocidad Alta Media (km)',
                gridcolor='rgb(220, 220, 220)'
            ),
            plot_bgcolor='rgb(250, 250, 250)',
            paper_bgcolor='rgb(250, 250, 250)',
            font=dict(color='rgb(20, 20, 20)'),
            barmode='group',
            bargap=0.15,
            bargroupgap=0.1,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(t=80, b=120, l=60, r=40),
            height=600
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error al visualizar las estadísticas de velocidad: {e}")
        return None

# Función principal
def main():
    # Agregar encabezado con logo y título
    add_header()
    
    # Agregar separador
    st.markdown("---")
    
    df = load_data()
    
    if df is not None:
       
        # Obtener lista de jugadores
        jugadores = obtener_jugadores(df)
        
        if jugadores:            # Crear selector de jugador
            jugador_seleccionado = st.selectbox(
                "Selecciona un jugador para ver sus estadísticas:",
                jugadores
            )
            
            # Procesar datos del jugador seleccionado
            datos_jugador = procesar_datos_jugador(df, jugador_seleccionado)
            
            if datos_jugador is not None and not datos_jugador.empty:
                st.info(f"Se encontraron {len(datos_jugador)} registros para {jugador_seleccionado}")
                
                # PANEL DE CARGA CON TRES COLUMNAS
                mostrar_panel_carga(datos_jugador, jugador_seleccionado)
                
                st.markdown("---")
                
                # Selector para cantidad de sesiones a mostrar en gráficos
                n_sesiones = st.slider(
                    "Número de sesiones a mostrar en gráficos",
                    min_value=5,
                    max_value=min(30, len(datos_jugador)),
                    value=10,
                    step=1
                )
                
               
                  # SECCIÓN 1: EVOLUCIÓN DE DISTANCIA
                st.markdown("## 📏 Evolución de Distancia")
                fig_distancia = graficar_distancia_jugador(datos_jugador, n_sesiones)
                if fig_distancia:
                    st.plotly_chart(fig_distancia, use_container_width=True)
                
                st.markdown("---")
                
                # SECCIÓN 2: EVOLUCIÓN DE VELOCIDAD ALTA
                st.markdown("## 🚀 Evolución de Velocidad Alta")
                fig_velocidad = graficar_velocidad_jugador(datos_jugador, n_sesiones)
                if fig_velocidad:
                    st.plotly_chart(fig_velocidad, use_container_width=True)
                
                st.markdown("---")
                
                # SECCIÓN 3: EVOLUCIÓN DE ACELERACIONES
                st.markdown("## 🔄 Evolución de Aceleraciones")
                fig_aceleraciones = graficar_aceleraciones_jugador(datos_jugador, n_sesiones)
                if fig_aceleraciones:
                    st.plotly_chart(fig_aceleraciones, use_container_width=True)
                
                st.markdown("---")
                
                # SECCIÓN 4: TABLA RESUMEN POR TIPO DE SESIÓN
                st.markdown("## 📋 Resumen Estadístico por Tipo de Sesión")
                resumen = crear_tabla_resumen(datos_jugador)
                if resumen is not None:
                    st.dataframe(resumen)
                
            else:
                st.warning(f"No se pudieron procesar datos para {jugador_seleccionado}")
        else:
            st.error("No se pudieron obtener nombres de jugadores de los datos")
    else:
        st.error("No se pudieron cargar los datos. Verifique la conexión o la URL del archivo.")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()

