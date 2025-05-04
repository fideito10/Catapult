import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Agregar la ruta del proyecto al path de Python
root_path = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_path))

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

# Configuración de la página
st.set_page_config(page_title="Universitario de La Plata - Dashboard Jugador", layout="wide")

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

# Función para calcular ratio Agudo:Crónico
def calcular_ratio_AC(datos_jugador, ventana_aguda=7, ventana_cronica=28):
    """
    Calcula el ratio de carga aguda:crónica basado en la distancia recorrida.
    
    Parámetros:
    - datos_jugador: DataFrame con los datos del jugador
    - ventana_aguda: Número de días para la carga aguda (por defecto 7 días)
    - ventana_cronica: Número de días para la carga crónica (por defecto 28 días)
    
    Retorna:
    - DataFrame con fechas, distancias, carga aguda, carga crónica y ratio A:C
    """
    if datos_jugador is None or datos_jugador.empty or 'Date' not in datos_jugador.columns:
        return None
    
    # Asegurar que la fecha está ordenada cronológicamente
    datos_ordenados = datos_jugador.sort_values('Date')
    
    # Crear lista para almacenar resultados
    resultados = []
    
    # Para cada fecha, calcular cargas y ratio
    fechas_unicas = datos_ordenados['Date'].unique()
    
    for i in range(ventana_cronica, len(fechas_unicas) + 1):
        fecha_actual = fechas_unicas[i-1]
        
        # Datos de la sesión actual
        datos_sesion_actual = datos_ordenados[datos_ordenados['Date'] == fecha_actual]
        distancia_sesion = datos_sesion_actual['Distance (km)'].sum()
        
        # Datos para ventana aguda (últimos 7 días)
        indice_inicio_agudo = max(0, i - ventana_aguda)
        fechas_agudo = fechas_unicas[indice_inicio_agudo:i]
        datos_agudo = datos_ordenados[datos_ordenados['Date'].isin(fechas_agudo)]
        
        # Datos para ventana crónica (últimos 28 días)
        indice_inicio_cronico = max(0, i - ventana_cronica)
        fechas_cronico = fechas_unicas[indice_inicio_cronico:i]
        datos_cronico = datos_ordenados[datos_ordenados['Date'].isin(fechas_cronico)]
        
        # Calcular carga aguda y crónica (promedios diarios)
        carga_aguda = datos_agudo['Distance (km)'].mean()
        carga_cronica = datos_cronico['Distance (km)'].mean()
        
        # Calcular ratio A:C
        ratio_ac = carga_aguda / carga_cronica if carga_cronica > 0 else 0
        
        # Guardar resultados
        resultados.append({
            'Fecha': fecha_actual,
            'Session Title': datos_sesion_actual['Session Title'].iloc[0],
            'Distancia (km)': distancia_sesion,
            'Carga Aguda': carga_aguda,
            'Carga Crónica': carga_cronica,
            'Ratio A:C': ratio_ac
        })
    
    # Convertir resultados a DataFrame
    if resultados:
        df_resultados = pd.DataFrame(resultados)
        # Ordenar por fecha descendente para mostrar los más recientes primero
        return df_resultados.sort_values('Fecha', ascending=False)
    else:
        return None

# Función para cargar datos
@st.cache_data
def load_data():
    sheet_id = "1kajUuZwL9l1suipRNy7t2g_SGJUcDVh6MeTu66yK-Z4"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    
    try:
        df = pd.read_csv(url)
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
        if 'Distance (km)' in datos_jugador.columns:
            if isinstance(datos_jugador['Distance (km)'].iloc[0], str):
                datos_jugador['Distance (km)'] = datos_jugador['Distance (km)'].str.replace(',', '.').astype(float)
            elif not pd.api.types.is_numeric_dtype(datos_jugador['Distance (km)']):
                datos_jugador['Distance (km)'] = pd.to_numeric(datos_jugador['Distance (km)'], errors='coerce')
        
        # Procesar métricas de velocidad alta
        columnas_velocidad = [
            'Distance in Speed Zone 3 (km)',
            'Distance in Speed Zone 4 (km)',
            'Distance in Speed Zone 5 (km)'
        ]
        
        for col in columnas_velocidad:
            if col in datos_jugador.columns:
                if isinstance(datos_jugador[col].iloc[0], str):
                    datos_jugador[col] = datos_jugador[col].str.replace(',', '.').astype(float)
                elif not pd.api.types.is_numeric_dtype(datos_jugador[col]):
                    datos_jugador[col] = pd.to_numeric(datos_jugador[col], errors='coerce')
        
        datos_jugador['Alta Velocidad (km)'] = datos_jugador[columnas_velocidad].sum(axis=1)
        
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
        
        for col in columnas_aceleraciones + columnas_desaceleraciones:
            if col in datos_jugador.columns:
                if isinstance(datos_jugador[col].iloc[0], str):
                    datos_jugador[col] = datos_jugador[col].str.replace(',', '.').astype(float)
                elif not pd.api.types.is_numeric_dtype(datos_jugador[col]):
                    datos_jugador[col] = pd.to_numeric(datos_jugador[col], errors='coerce')
        
        datos_jugador['Total Aceleraciones'] = datos_jugador[columnas_aceleraciones].sum(axis=1)
        datos_jugador['Total Desaceleraciones'] = datos_jugador[columnas_desaceleraciones].sum(axis=1)
        
        # Ordenar por fecha
        datos_jugador = datos_jugador.sort_values('Date', ascending=True)
        
        return datos_jugador
    
    except Exception as e:
        st.error(f"Error al procesar datos del jugador: {e}")
        return None

# Función para graficar la evolución de distancia por sesión para un jugador
def graficar_distancia_jugador(datos_jugador, n_sesiones=10):
    if datos_jugador is None or datos_jugador.empty:
        st.warning("No hay datos para visualizar")
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
    
    # Crear figura con Plotly
    fig = go.Figure()
    
    # Añadir barras para distancia
    fig.add_trace(go.Bar(
        x=datos_plot['Date'].dt.date,
        y=datos_plot['Distance (km)'],
        name='Distancia',
        marker_color='rgb(70, 70, 70)',
        text=datos_plot['Distance (km)'].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Sesión: %{customdata}<br>Distancia: %{y:.2f} km<extra></extra>',
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
    
    # Actualizar diseño
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
            title_font=dict(size=16)
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
        height=500
    )
    
    return fig

# Función para graficar la evolución de velocidad alta por sesión para un jugador
def graficar_velocidad_jugador(datos_jugador, n_sesiones=10):
    if datos_jugador is None or datos_jugador.empty or 'Alta Velocidad (km)' not in datos_jugador.columns:
        st.warning("No hay datos de velocidad alta para visualizar")
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
    
    # Crear figura con Plotly
    fig = go.Figure()
    
    # Añadir barras para velocidad alta
    fig.add_trace(go.Bar(
        x=datos_plot['Date'].dt.date,
        y=datos_plot['Alta Velocidad (km)'],
        name='Velocidad Alta',
        marker_color='rgb(65, 105, 225)',  # Azul royal
        text=datos_plot['Alta Velocidad (km)'].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Sesión: %{customdata}<br>Velocidad Alta: %{y:.2f} km<extra></extra>',
        customdata=datos_plot['Session Title']
    ))
    
    # Agregar línea de tendencia
    fig.add_trace(go.Scatter(
        x=datos_plot['Date'].dt.date,
        y=datos_plot['Alta Velocidad (km)'],
        name='Tendencia',
        mode='lines',
        line=dict(color='rgb(30, 30, 30)', width=3, shape='spline'),
        hovertemplate='<b>%{x}</b><br>Tendencia: %{y:.2f} km<extra></extra>'
    ))
    
    # Actualizar diseño
    fig.update_layout(
        title={
            'text': 'Evolución de Velocidad Alta por Sesión',
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Fecha',
            tickangle=-30,
            tickfont=dict(size=14),
            title_font=dict(size=16)
        ),
        yaxis=dict(
            title='Velocidad Alta (km)',
            gridcolor='rgb(220, 220, 220)'
        ),
        plot_bgcolor='rgb(250, 250, 250)',
        paper_bgcolor='rgb(250, 250, 250)',
        font=dict(color='rgb(20, 20, 20)'),
        bargap=0.2,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    
    return fig

# Función para graficar la evolución de aceleraciones por sesión para un jugador
def graficar_aceleraciones_jugador(datos_jugador, n_sesiones=10):
    if datos_jugador is None or datos_jugador.empty:
        st.warning("No hay datos de aceleraciones para visualizar")
        return None
    
    if 'Total Aceleraciones' not in datos_jugador.columns or 'Total Desaceleraciones' not in datos_jugador.columns:
        st.warning("No se encontraron datos de aceleraciones o desaceleraciones")
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
    
    # Crear figura con Plotly
    fig = go.Figure()
    
    # Añadir barras para aceleraciones
    fig.add_trace(go.Bar(
        x=datos_plot['Date'].dt.date,
        y=datos_plot['Total Aceleraciones'],
        name='Aceleraciones',
        marker_color='rgb(65, 105, 225)',  # Azul royal
        text=datos_plot['Total Aceleraciones'].round(0).astype(int),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Sesión: %{customdata}<br>Aceleraciones: %{y}<extra></extra>',
        customdata=datos_plot['Session Title']
    ))
    
    # Añadir barras para desaceleraciones
    fig.add_trace(go.Bar(
        x=datos_plot['Date'].dt.date,
        y=datos_plot['Total Desaceleraciones'],
        name='Desaceleraciones',
        marker_color='rgb(220, 20, 60)',  # Rojo carmesí
        text=datos_plot['Total Desaceleraciones'].round(0).astype(int),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Sesión: %{customdata}<br>Desaceleraciones: %{y}<extra></extra>',
        customdata=datos_plot['Session Title']
    ))
    
    # Actualizar diseño
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
            title_font=dict(size=16)
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
        height=500
    )
    
    return fig

# Función para crear una tabla de resumen de estadísticas por tipo de sesión
def crear_tabla_resumen(datos_jugador):
    if datos_jugador is None or datos_jugador.empty:
        st.warning("No hay datos suficientes para mostrar estadísticas")
        return None
    
    # Verificar columnas necesarias
    columnas_requeridas = ['Session Title', 'Distance (km)', 'Alta Velocidad (km)', 
                          'Total Aceleraciones', 'Total Desaceleraciones']
    
    if not all(col in datos_jugador.columns for col in columnas_requeridas):
        st.warning("Faltan algunas columnas necesarias para las estadísticas")
        return None
    
    # Agrupar por tipo de sesión y calcular estadísticas
    resumen = datos_jugador.groupby('Session Title').agg({
        'Distance (km)': ['mean', 'min', 'max', 'count'],
        'Alta Velocidad (km)': ['mean', 'min', 'max'],
        'Total Aceleraciones': ['mean', 'min', 'max'],
        'Total Desaceleraciones': ['mean', 'min', 'max']
    }).reset_index()
    
    # Aplanar MultiIndex en columnas
    resumen.columns = [' '.join(col).strip() for col in resumen.columns.values]
    
    # Renombrar columnas para mejor visualización
    resumen = resumen.rename(columns={
        'Session Title': 'Tipo de Sesión',
        'Distance (km) mean': 'Distancia Media (km)',
        'Distance (km) min': 'Distancia Mínima (km)',
        'Distance (km) max': 'Distancia Máxima (km)',
        'Distance (km) count': 'Cantidad de Sesiones',
        'Alta Velocidad (km) mean': 'Velocidad Alta Media (km)',
        'Alta Velocidad (km) min': 'Velocidad Alta Mínima (km)',
        'Alta Velocidad (km) max': 'Velocidad Alta Máxima (km)',
        'Total Aceleraciones mean': 'Aceleraciones Promedio',
        'Total Aceleraciones min': 'Aceleraciones Mínimas',
        'Total Aceleraciones max': 'Aceleraciones Máximas',
        'Total Desaceleraciones mean': 'Desaceleraciones Promedio',
        'Total Desaceleraciones min': 'Desaceleraciones Mínimas',
        'Total Desaceleraciones max': 'Desaceleraciones Máximas'
    })
    
    # Redondear valores numéricos para mejor visualización
    columnas_numericas = resumen.columns.drop('Tipo de Sesión')
    for col in columnas_numericas:
        if 'Cantidad' in col:
            resumen[col] = resumen[col].astype(int)
        elif 'Aceleraciones' in col or 'Desaceleraciones' in col:
            resumen[col] = resumen[col].round(0).astype(int)
        else:
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
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        df = load_data()
    
    if df is not None:
        st.success(f"Datos cargados correctamente: {len(df)} registros")
        
        # Obtener lista de jugadores
        jugadores = obtener_jugadores(df)
        
        if jugadores:
            # Crear selector de jugador
            jugador_seleccionado = st.selectbox(
                "Selecciona un jugador para ver sus estadísticas:",
                jugadores
            )
            
            # Procesar datos del jugador seleccionado
            datos_jugador = procesar_datos_jugador(df, jugador_seleccionado)
            
            if datos_jugador is not None and not datos_jugador.empty:
                st.info(f"Se encontraron {len(datos_jugador)} registros para {jugador_seleccionado}")
                
                # Selector para cantidad de sesiones a mostrar en gráficos
                n_sesiones = st.slider(
                    "Número de sesiones a mostrar en gráficos",
                    min_value=5,
                    max_value=min(30, len(datos_jugador)),
                    value=10,
                    step=1
                )
                
                # Mostrar métricas recientes
                st.markdown("## 📊 Métricas Recientes (Últimos 30 días)")
                mostrar_metricas_recientes(datos_jugador)
                
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