import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from PIL import Image


# Configuración de la página
st.set_page_config(page_title="Universitario de La Plata - Dashboard", layout="wide")

# Función para agregar logo y título
def add_header():
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.title("Universitario de La Plata")
        st.markdown("### Dashboard de Análisis de Rendimiento")
    
    with col2:
        # Puedes reemplazar esto con la URL del logo real o un archivo local
        try:
            logo = Image.open(r"C:\Users\dell\Desktop\Python\Catapult\escudo uni.jpg")
            st.image(logo, width=450)
        except Exception:
            # Si no se puede cargar el logo, crear uno simple
            st.warning("No se pudo cargar el logo")

# Función para cargar datos
@st.cache_data
def load_data():
    sheet_id = "1kajUuZwL9l1suipRNy7t2g_SGJUcDVh6MeTu66yK-Z4"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    
    try:
        # Leer directamente como DataFrame
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

def procesar_sesiones(df, tipo_sesion='Partido', max_fechas=None):
    """
    Procesa datos de sesiones específicas (Partido, Martes o Jueves)
    
    Parámetros:
    - df: DataFrame con los datos
    - tipo_sesion: Tipo de sesión a procesar ('Partido', 'Martes', 'Jueves')
    - max_fechas: Número máximo de fechas a mostrar (None para mostrar todas)
    """
    # Verificar si el DataFrame es válido
    if df is None or df.empty:
        st.warning("No hay datos disponibles para analizar.")
        return None
    
    # Filtrar solo los registros que corresponden al tipo de sesión
    sesiones_df = df[df['Session Title'].str.contains(tipo_sesion, case=False, na=False)].copy()
    
    if sesiones_df.empty:
        st.warning(f"No se encontraron registros de {tipo_sesion} en los datos.")
        return None
    
    # Convertir columnas necesarias
    try:
        # Convertir fecha a datetime si no lo está
        if not pd.api.types.is_datetime64_dtype(sesiones_df['Date']):
            sesiones_df['Date'] = pd.to_datetime(sesiones_df['Date'], errors='coerce')
        
        # Convertir distancia a numérica
        if isinstance(sesiones_df['Distance (km)'].iloc[0], str):
            sesiones_df['Distance (km)'] = sesiones_df['Distance (km)'].str.replace(',', '.').astype(float)
        elif not pd.api.types.is_numeric_dtype(sesiones_df['Distance (km)']):
            sesiones_df['Distance (km)'] = pd.to_numeric(sesiones_df['Distance (km)'], errors='coerce')
    
    except Exception as e:
        st.error(f"Error al convertir datos: {e}")
        return None
    
    # Agrupar por fecha y obtener estadísticas
    sesiones_stats = sesiones_df.groupby([sesiones_df['Date'].dt.date, 'Session Title'])['Distance (km)'].agg(
        ['mean', 'count', 'sum']).reset_index()
    
    sesiones_stats.columns = ['Fecha', 'Sesión', 'Distancia Media (km)', 
                              'N° Jugadores', 'Distancia Total (km)']
    
    # Ordenar por fecha ascendente
    sesiones_stats = sesiones_stats.sort_values('Fecha', ascending=True)
    
    # Crear una columna de identificación única que combine fecha y sesión
    sesiones_stats['Fecha_Sesion'] = sesiones_stats['Sesión']  # Solo usar el título de la sesión.astype(str) + ' - ' + sesiones_stats['Sesión']
    
    # Limitar por cantidad de fechas si se especifica
    if max_fechas and max_fechas > 0 and max_fechas < len(sesiones_stats):
        # Tomar los últimos N registros (los más recientes)
        sesiones_stats = sesiones_stats.tail(max_fechas)
    
    return sesiones_stats

def graficar_distancia_sesion(sesiones_stats, n_sesiones, tipo_sesion):
    """
    Genera el gráfico de distancia por sesiones
    """
    if sesiones_stats is None or sesiones_stats.empty:
        st.warning("No hay datos para visualizar")
        return
    
    # Limitar cantidad de sesiones a mostrar
    # Tomamos las últimas n_sesiones (más recientes)
    df_plot = sesiones_stats.tail(n_sesiones).copy()
    
    # Crear figura con Plotly
    fig = go.Figure()
    
    # Añadir barras para distancia media
    fig.add_trace(go.Bar(
        x=df_plot['Fecha_Sesion'],
        y=df_plot['Distancia Media (km)'],
        name='Distancia Media',
        marker_color='rgb(70, 70, 70)',
        text=df_plot['Distancia Media (km)'].round(2),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Distancia Media: %{y:.2f} km<br>N° Jugadores: %{customdata[0]}<br>Total: %{customdata[1]:.2f} km<extra></extra>',
        customdata=df_plot[['N° Jugadores', 'Distancia Total (km)']]
    ))
    
    # Agregar línea de tendencia
    fig.add_trace(go.Scatter(
        x=df_plot['Fecha_Sesion'],
        y=df_plot['Distancia Media (km)'],
        name='Tendencia',
        mode='lines',
        line=dict(color='rgb(30, 30, 30)', width=3, shape='spline'),
        hovertemplate='<b>%{x}</b><br>Tendencia: %{y:.2f} km<extra></extra>'
    ))
    
    # Actualizar diseño
    fig.update_layout(
    title={
        'text': f'Distancia en (km) - {tipo_sesion}',
        'font': {'size': 24},
        'x': 0.5,
        'xanchor': 'center'
    },
    xaxis=dict(
        title='Sesión',
        tickangle=-30,  # Ángulo menos pronunciado
        tickfont=dict(size=14),  # Texto más grande
        title_font=dict(size=16)  # Título más grande
    ),
    yaxis=dict(
        title='Distancia (km)',
        gridcolor='rgb(220, 220, 220)'
    ),
        plot_bgcolor='rgb(250, 250, 250)',
        paper_bgcolor='rgb(250, 250, 250)',
        font=dict(color='rgb(20, 20, 20)'),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    
    return fig

def graficar_velocidad_sesion(velocidad_stats, tipo_sesion):
    """
    Genera el gráfico de velocidad alta por sesiones
    """
    if velocidad_stats is None or velocidad_stats.empty:
        st.warning("No hay datos de velocidad para visualizar")
        return
    
    # Ya no es necesario crear ID_Sesion porque usamos directamente 'Partido'
    # que ya está agrupado correctamente
    
    # Limitar a las últimas 10 sesiones para mejor visualización
    if len(velocidad_stats) > 10:
        velocidad_stats = velocidad_stats.tail(10)
    
    # Crear gráfico de barras con Plotly
    fig = go.Figure()
    
    # Añadir barras para la velocidad alta media
    fig.add_trace(go.Bar(
        x=velocidad_stats['Partido'],  # Usar directamente 'Partido' en lugar de 'ID_Sesion'
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
        x=velocidad_stats['Partido'],  # Usar directamente 'Partido' en lugar de 'ID_Sesion'
        y=velocidad_stats['Velocidad Alta Media (km)'],
        name='Tendencia',
        mode='lines+markers',
        line=dict(color='rgb(30, 30, 30)', width=2),
        marker=dict(color='rgb(30, 30, 30)', size=8),
        hoverinfo='skip'
    ))
      # Actualizar diseño (falta esta parte en tu código)
    fig.update_layout(
        title={
            'text': f'Velocidad Alta (km) - {tipo_sesion}',
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Sesión',
            tickangle=-30,
            tickfont=dict(size=14),
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
        height=500
    )
    
    # Añadir return fig (falta en tu código)
    return fig
 

def obtener_estadisticas_velocidad(sesiones_df):

    try:
        # Columnas de velocidad alta
        columnas_velocidad = [
            'Distance in Speed Zone 3 (km)',
            'Distance in Speed Zone 4 (km)',
            'Distance in Speed Zone 5 (km)'
        ]
        
        # Convertir las columnas a numéricas si es necesario
        for col in columnas_velocidad:
            if isinstance(sesiones_df[col].iloc[0], str):
                sesiones_df[col] = sesiones_df[col].str.replace(',', '.').astype(float)
            elif not pd.api.types.is_numeric_dtype(sesiones_df[col]):
                sesiones_df[col] = pd.to_numeric(sesiones_df[col], errors='coerce')
        
        # Crear columna con la suma de las zonas de velocidad alta
        sesiones_df['Alta Velocidad (km)'] = sesiones_df[columnas_velocidad].sum(axis=1)
        
        # Cambio importante: agrupar solo por Session Title, no por fecha y Session Title
        velocidad_stats = sesiones_df.groupby(['Session Title'])['Alta Velocidad (km)'].agg(
            ['mean', 'std', 'count', 'sum']).reset_index()
        
        # Renombrar columnas
        velocidad_stats.columns = ['Partido', 'Velocidad Alta Media (km)', 
                                  'Desviación Estándar (km)', 'N° Jugadores', 'Velocidad Alta Total (km)']
        
        # Ordenar alfabéticamente por el nombre del partido/sesión
        velocidad_stats = velocidad_stats.sort_values('Partido', ascending=True)
        
        return velocidad_stats
        
    except Exception as e:
        st.error(f"Error al obtener estadísticas de velocidad: {e}")
        return None
    
    
def obtener_estadisticas_aceleraciones(sesiones_df):
   
    try:
        # Columnas de aceleraciones
        columnas_aceleraciones = [
            'Accelerations Zone Count: 2 - 3 m/s/s', 
            'Accelerations Zone Count: 3 - 4 m/s/s', 
            'Accelerations Zone Count: > 4 m/s/s'
        ]
        
        # Columnas de desaceleraciones
        columnas_desaceleraciones = [
            'Deceleration Zone Count: 1 - 2 m/s/s', 
            'Deceleration Zone Count: 2 - 3 m/s/s', 
            'Deceleration Zone Count: 3 - 4 m/s/s', 
            'Deceleration Zone Count: > 4 m/s/s'
        ]
        
        # Convertir columnas a numéricas si es necesario
        for col in columnas_aceleraciones + columnas_desaceleraciones:
            if isinstance(sesiones_df[col].iloc[0], str):
                sesiones_df[col] = sesiones_df[col].str.replace(',', '.').astype(float)
            elif not pd.api.types.is_numeric_dtype(sesiones_df[col]):
                sesiones_df[col] = pd.to_numeric(sesiones_df[col], errors='coerce')
        
        # Crear columnas con la suma de aceleraciones y desaceleraciones
        sesiones_df['Total Aceleraciones'] = sesiones_df[columnas_aceleraciones].sum(axis=1)
        sesiones_df['Total Desaceleraciones'] = sesiones_df[columnas_desaceleraciones].sum(axis=1)
        
        # Agrupar por Session Title
        aceleraciones_stats = sesiones_df.groupby(['Session Title'])[
            ['Total Aceleraciones', 'Total Desaceleraciones']].agg(['mean', 'count']).reset_index()
        
        # Reorganizar columnas para facilitar su uso
        aceleraciones_stats.columns = [
            'Sesión',
            'Media Aceleraciones', 'Count Aceleraciones',
            'Media Desaceleraciones', 'Count Desaceleraciones'
        ]
        
        # Ordenar alfabéticamente por el nombre de la sesión
        aceleraciones_stats = aceleraciones_stats.sort_values('Sesión', ascending=True)
        
        return aceleraciones_stats
        
    except Exception as e:
        print(f"Error al obtener estadísticas de aceleraciones: {e}")
        return None

def graficar_aceleraciones(aceleraciones_stats, tipo_sesion, n_sesiones=10):
    """
    Genera un gráfico de barras comparando aceleraciones (azul) y desaceleraciones (rojo)
    """
    import plotly.graph_objects as go
    
    if aceleraciones_stats is None or aceleraciones_stats.empty:
        print("No hay datos de aceleraciones para visualizar")
        return None
    
    # Limitar a las últimas n sesiones para mejor visualización
    if len(aceleraciones_stats) > n_sesiones:
        aceleraciones_stats = aceleraciones_stats.tail(n_sesiones)
    
    # Crear figura con Plotly
    fig = go.Figure()
    
    # Añadir barras para aceleraciones (en azul)
    fig.add_trace(go.Bar(
        x=aceleraciones_stats['Sesión'],
        y=aceleraciones_stats['Media Aceleraciones'],
        name='Aceleraciones',
        marker_color='rgb(65, 105, 225)',  # Azul royal
        text=aceleraciones_stats['Media Aceleraciones'].round(1),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Aceleraciones: %{y:.1f}<br>Conteo: %{customdata}<extra></extra>',
        customdata=aceleraciones_stats['Count Aceleraciones']
    ))
    
    # Añadir barras para desaceleraciones (en rojo)
    fig.add_trace(go.Bar(
        x=aceleraciones_stats['Sesión'],
        y=aceleraciones_stats['Media Desaceleraciones'],
        name='Desaceleraciones',
        marker_color='rgb(220, 20, 60)',  # Rojo carmesí
        text=aceleraciones_stats['Media Desaceleraciones'].round(1),
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Desaceleraciones: %{y:.1f}<br>Conteo: %{customdata}<extra></extra>',
        customdata=aceleraciones_stats['Count Desaceleraciones']
    ))
    
    # Actualizar diseño
    fig.update_layout(
        title={
            'text': f'Aceleraciones vs Desaceleraciones - {tipo_sesion}',
            'font': {'size': 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title='Sesión',
            tickangle=-30,
            tickfont=dict(size=14),
            title_font=dict(size=16)
        ),
        yaxis=dict(
            title='Cantidad promedio',
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

def graficar_velocidades_maximas(df_ultima_sesion, titulo_sesion):
    """
    Genera un gráfico de barras ordenado con las velocidades máximas de los jugadores
    
    Parámetros:
    - df_ultima_sesion: DataFrame con los datos de la última sesión
    - titulo_sesion: Título de la sesión (ej: "Partido", "Martes", etc.)
    
    Retorna:
    - fig: Objeto de figura de Plotly con el gráfico
    """
    if df_ultima_sesion is None or df_ultima_sesion.empty:
        return None
    
    try:
        # Verificar si existe la columna de velocidad máxima
        if 'Top Speed (m/s)' not in df_ultima_sesion.columns:
            print("No se encontró la columna 'Top Speed (m/s)'")
            return None
        
        # Convertir a numérico si es necesario
        if isinstance(df_ultima_sesion['Top Speed (m/s)'].iloc[0], str):
            df_ultima_sesion['Top Speed (m/s)'] = df_ultima_sesion['Top Speed (m/s)'].str.replace(',', '.').astype(float)
        elif not pd.api.types.is_numeric_dtype(df_ultima_sesion['Top Speed (m/s)']):
            df_ultima_sesion['Top Speed (m/s)'] = pd.to_numeric(df_ultima_sesion['Top Speed (m/s)'], errors='coerce')
        
        # Ordenar por velocidad ascendente
        velocidades_ordenadas = df_ultima_sesion[['Player Name', 'Top Speed (m/s)']] \
            .sort_values('Top Speed (m/s)', ascending=True)
        
        # Crear gráfico con Plotly
        fig = go.Figure()
        
        # Añadir barras para velocidades
        fig.add_trace(go.Bar(
            x=velocidades_ordenadas['Player Name'],
            y=velocidades_ordenadas['Top Speed (m/s)'],
            text=velocidades_ordenadas['Top Speed (m/s)'].round(1),
            textposition='outside',
            marker_color='rgb(70, 70, 70)',
            hovertemplate='<b>%{x}</b><br>Velocidad Máxima: %{y:.1f} m/s<extra></extra>'
        ))
        
        # Actualizar diseño
        fig.update_layout(
            title={
                'text': f'Velocidades Máximas - {titulo_sesion}',
                'font': {'size': 24},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis=dict(
                title='Jugador',
                tickangle=-45,
                tickfont=dict(size=12),
                title_font=dict(size=16)
            ),
            yaxis=dict(
                title='Velocidad Máxima (m/s)',
                gridcolor='rgb(220, 220, 220)'
            ),
            plot_bgcolor='rgb(250, 250, 250)',
            paper_bgcolor='rgb(250, 250, 250)',
            showlegend=False,
            height=600
        )
        
        return fig
        
    except Exception as e:
        print(f"Error al generar gráfico de velocidades máximas: {e}")
        return None

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
        
        # Crear selector para tipo de sesión
        tipos_sesion = ["Partido", "Martes", "Jueves"]
        tipo_seleccionado = st.selectbox(
            "Selecciona el tipo de sesión a analizar:",
            tipos_sesion
        )
        
        # Procesar datos según el tipo de sesión seleccionado
        sesiones_stats = procesar_sesiones(df, tipo_sesion=tipo_seleccionado)
        
        if sesiones_stats is not None and not sesiones_stats.empty:
            st.info(f"Se encontraron {len(sesiones_stats)} registros de {tipo_seleccionado}")
            
            # Crear selector para número de sesiones
            n_sesiones = st.slider(
                f"Número de {tipo_seleccionado.lower()}s a mostrar",
                min_value=1,
                max_value=min(20, len(sesiones_stats)),
                value=5,
                step=1
            )
            
            # Procesar datos de sesiones, velocidad y aceleraciones
            sesiones_df = df[df['Session Title'].str.contains(tipo_seleccionado, case=False, na=False)].copy()
            
            # Convertir fecha a datetime si no lo está
            if not pd.api.types.is_datetime64_dtype(sesiones_df['Date']):
                sesiones_df['Date'] = pd.to_datetime(sesiones_df['Date'], errors='coerce')
            
            # Procesar datos de última sesión para los Top 3
            try:
                ultima_fecha = sesiones_df['Date'].max()
                ultima_sesion = sesiones_df[sesiones_df['Date'] == ultima_fecha]
            except Exception as e:
                st.error(f"Error al procesar fecha más reciente: {e}")
                ultima_sesion = None
            
            # SECCIÓN 1: DISTANCIA + TOP 3 DISTANCIA
            st.markdown("## 📊 Análisis de Distancia")
            col_dist_graf, col_dist_tabla = st.columns([7, 3])
            
            # Gráfico de Distancia
            with col_dist_graf:
                with st.container():
                    fig_distancia = graficar_distancia_sesion(sesiones_stats, n_sesiones, tipo_seleccionado)
                    if fig_distancia:
                        st.plotly_chart(fig_distancia, use_container_width=True)
                    else:
                        st.warning(f"No se pudieron generar gráficos de distancia para {tipo_seleccionado}")
            
            # Top 3 Distancia
            with col_dist_tabla:
                st.markdown("### 🥇 Top 3 Distancia")
                try:
                    if ultima_sesion is not None and not ultima_sesion.empty:
                        # Top 3 por distancia
                        top_distancia = ultima_sesion.sort_values('Distance (km)', ascending=False).head(3)
                        top_distancia_tabla = top_distancia[['Player Name', 'Distance (km)']]
                        top_distancia_tabla['Distance (km)'] = top_distancia_tabla['Distance (km)'].round(2)
                        top_distancia_tabla.columns = ['Nombre', 'Distancia (km)']
                        top_distancia_tabla['Distancia (km)'] = top_distancia_tabla['Distancia (km)'].astype(float)
                        st.table(top_distancia_tabla)
                    else:
                        st.warning("No hay datos de la última sesión")
                except Exception as e:
                    st.error(f"Error al procesar datos de distancia: {e}")
            
            st.markdown("---")
            
            # SECCIÓN 2: VELOCIDAD ALTA + TOP 3 VELOCIDAD ALTA
            st.markdown("## 🚀 Análisis de Velocidad Alta")
            col_vel_graf, col_vel_tabla = st.columns([7, 3])
            
            # Gráfico de Velocidad Alta
            with col_vel_graf:
                with st.spinner(f"Analizando datos de velocidad para {tipo_seleccionado}..."):
                    velocidad_stats = obtener_estadisticas_velocidad(sesiones_df)
                    if velocidad_stats is not None:
                        fig_velocidad = graficar_velocidad_sesion(velocidad_stats, tipo_seleccionado)
                        if fig_velocidad:
                            st.plotly_chart(fig_velocidad, use_container_width=True)
                        else:
                            st.warning(f"No se pudieron generar gráficos de velocidad para {tipo_seleccionado}")
                    else:
                        st.warning(f"No se pudieron obtener estadísticas de velocidad para {tipo_seleccionado}")
            
            # Top 3 Velocidad Alta
            with col_vel_tabla:
                st.markdown("### 🥇 Top 3 Velocidad Alta")
                try:
                    if ultima_sesion is not None and not ultima_sesion.empty:
                        if 'Alta Velocidad (km)' in ultima_sesion.columns:
                            # Top 3 por velocidad alta
                            top_velocidad = ultima_sesion.sort_values('Alta Velocidad (km)', ascending=False).head(3)
                            top_velocidad_tabla = top_velocidad[['Player Name', 'Alta Velocidad (km)']]
                            top_velocidad_tabla['Alta Velocidad (km)'] = top_velocidad_tabla['Alta Velocidad (km)'].round(2)
                            # Corrección de los nombres de columnas
                            top_velocidad_tabla.columns = ['Nombre', 'Velocidad Alta (km)']
                            st.table(top_velocidad_tabla)
                        else:
                            # Necesitamos calcular la velocidad alta primero
                            columnas_velocidad = [
                                'Distance in Speed Zone 3 (km)',
                                'Distance in Speed Zone 4 (km)',
                                'Distance in Speed Zone 5 (km)'
                            ]
                            
                            # Verificar si tenemos las columnas necesarias
                            if all(col in ultima_sesion.columns for col in columnas_velocidad):
                                # Convertir columnas a numéricas
                                for col in columnas_velocidad:
                                    if isinstance(ultima_sesion[col].iloc[0], str):
                                        ultima_sesion[col] = ultima_sesion[col].str.replace(',', '.').astype(float)
                                    elif not pd.api.types.is_numeric_dtype(ultima_sesion[col]):
                                        ultima_sesion[col] = pd.to_numeric(ultima_sesion[col], errors='coerce')
                                
                                # Calcular velocidad alta
                                ultima_sesion['Alta Velocidad (km)'] = ultima_sesion[columnas_velocidad].sum(axis=1)
                                
                                # Top 3 por velocidad alta
                                top_velocidad = ultima_sesion.sort_values('Alta Velocidad (km)', ascending=False).head(3)
                                # Aquí hay que verificar qué columnas existen realmente en tu dataset
                                # Si existe Player Name, es mejor usarla directamente
                                if 'Player Name' in ultima_sesion.columns:
                                    top_velocidad_tabla = top_velocidad[['Player Name', 'Alta Velocidad (km)']]
                                    top_velocidad_tabla['Alta Velocidad (km)'] = top_velocidad_tabla['Alta Velocidad (km)'].round(2)
                                    top_velocidad_tabla.columns = ['Nombre', 'Velocidad Alta (km)']
                                else:
                                    # Si no existe Player Name, usar Given Name y Family Name
                                    top_velocidad_tabla = top_velocidad[['Given Name', 'Family Name', 'Alta Velocidad (km)']]
                                    top_velocidad_tabla['Alta Velocidad (km)'] = top_velocidad_tabla['Alta Velocidad (km)'].round(2)
                                    top_velocidad_tabla.columns = ['Nombre', 'Apellido', 'Velocidad Alta (km)']
                                
                                st.table(top_velocidad_tabla)
                            else:
                                st.warning("No se encontraron datos de velocidad alta")
                    else:
                        st.warning("No hay datos de la última sesión")
                except Exception as e:
                    st.error(f"Error al procesar datos de velocidad alta: {e}")
            
            st.markdown("---")
            
            # SECCIÓN 3: ACELERACIONES + TOP 3 ACELERACIONES
            st.markdown("## 🔄 Análisis de Aceleraciones")
            col_acel_graf, col_acel_tabla = st.columns([7, 3])
            
            # Gráfico de Aceleraciones
            with col_acel_graf:
                with st.spinner(f"Analizando datos de aceleraciones para {tipo_seleccionado}..."):
                    aceleraciones_stats = obtener_estadisticas_aceleraciones(sesiones_df)
                    if aceleraciones_stats is not None:
                        fig_aceleraciones = graficar_aceleraciones(aceleraciones_stats, tipo_seleccionado)
                        if fig_aceleraciones:
                            st.plotly_chart(fig_aceleraciones, use_container_width=True)
                        else:
                            st.warning(f"No se pudieron generar gráficos de aceleraciones para {tipo_seleccionado}")
                    else:
                        st.warning(f"No se pudieron obtener estadísticas de aceleraciones para {tipo_seleccionado}")
            
            # Top 3 Aceleraciones
            with col_acel_tabla:
                st.markdown("### 🥇 Top 3 Aceleraciones")
                try:
                    if ultima_sesion is not None and not ultima_sesion.empty:
                        # Columnas de aceleraciones
                        columnas_aceleraciones = [
                            'Accelerations Zone Count: 2 - 3 m/s/s', 
                            'Accelerations Zone Count: 3 - 4 m/s/s', 
                            'Accelerations Zone Count: > 4 m/s/s'
                        ]
                        
                        # Verificar si tenemos las columnas necesarias
                        if all(col in ultima_sesion.columns for col in columnas_aceleraciones):
                            # Convertir columnas a numéricas
                            for col in columnas_aceleraciones:
                                if isinstance(ultima_sesion[col].iloc[0], str):
                                    ultima_sesion[col] = ultima_sesion[col].str.replace(',', '.').astype(float)
                                elif not pd.api.types.is_numeric_dtype(ultima_sesion[col]):
                                    ultima_sesion[col] = pd.to_numeric(ultima_sesion[col], errors='coerce')
                            
                            # Calcular aceleraciones totales
                            ultima_sesion['Total Aceleraciones'] = ultima_sesion[columnas_aceleraciones].sum(axis=1)
                            
                            # Top 3 por aceleraciones
                            top_acel = ultima_sesion.sort_values('Total Aceleraciones', ascending=False).head(3)
                            
                            # Verificar si existe Player Name o usar Given Name y Family Name
                            if 'Player Name' in ultima_sesion.columns:
                                top_acel_tabla = top_acel[['Player Name', 'Total Aceleraciones']]
                                top_acel_tabla['Total Aceleraciones'] = top_acel_tabla['Total Aceleraciones'].round(0).astype(int)
                                top_acel_tabla.columns = ['Nombre', 'Aceleraciones']
                            else:
                                top_acel_tabla = top_acel[['Given Name', 'Family Name', 'Total Aceleraciones']]
                                top_acel_tabla['Total Aceleraciones'] = top_acel_tabla['Total Aceleraciones'].round(0).astype(int)
                                top_acel_tabla.columns = ['Nombre', 'Apellido', 'Aceleraciones']
                            
                            st.table(top_acel_tabla)
                        else:
                            st.warning("No se encontraron datos de aceleraciones")
                    else:
                        st.warning("No hay datos de la última sesión")
                except Exception as e:
                    st.error(f"Error al procesar datos de aceleraciones: {e}")

            st.markdown("---")
            
            # SECCIÓN 4: VELOCIDADES MÁXIMAS
            st.markdown("## 🏃 Velocidades Máximas por Jugador")
            try:
                if ultima_sesion is not None and not ultima_sesion.empty:
                    fig_velocidades = graficar_velocidades_maximas(ultima_sesion, tipo_seleccionado)
                    if fig_velocidades:
                        st.plotly_chart(fig_velocidades, use_container_width=True)
                    else:
                        st.warning("No se pudieron generar gráficos de velocidades máximas")
                else:
                    st.warning("No hay datos de la última sesión para mostrar velocidades máximas")
            except Exception as e:
                st.error(f"Error al procesar datos de velocidades máximas: {e}")
        else:
            st.warning(f"No se encontraron datos de {tipo_seleccionado} para visualizar")
    else:
        st.error("No se pudieron cargar los datos. Verifique la conexión o la URL del archivo.")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
    # Ejecutar la función principal