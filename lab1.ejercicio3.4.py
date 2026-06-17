import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy.interpolate import interp1d
import math

# Configuración de la página
st.set_page_config(
    page_title="Simulador Bird 2B.6 - Flujo en Película Cilíndrica",
    page_icon="🌊",
    layout="wide"
)

# --- Inyección de CSS mejorada para modo oscuro premium ---
st.markdown("""
<style>
    /* Fondo general oscuro */
    .stApp {
        background-color: #0a0a0a !important;
    }
    
    /* Sidebar oscuro */
    .css-1d391kg, .st-emotion-cache-1d391kg {
        background-color: #141414 !important;
    }
    
    /* Títulos en blanco puro */
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Texto normal en gris claro */
    p, li, .stMarkdown, .stCaption {
        color: #cccccc !important;
    }
    
    /* Métricas - VALORES en cian neón */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        border: 1px solid #00f0ff !important;
        border-radius: 10px !important;
        padding: 15px !important;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.05) !important;
    }
    
    div[data-testid="metric-container"] label {
        color: #b0b0b0 !important;
        font-size: 14px !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #00f0ff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        text-shadow: 0 0 30px rgba(0, 240, 255, 0.2) !important;
    }
    
    /* Sliders en cian */
    .stSlider .stSliderLabel {
        color: #ffffff !important;
    }
    
    .stSlider div[data-baseweb="slider"] div[role="slider"] {
        background-color: #00f0ff !important;
        border-color: #00f0ff !important;
    }
    
    .stSlider div[data-baseweb="slider"] div[data-testid="track"] {
        background-color: #00f0ff !important;
    }
    
    /* Number inputs */
    .stNumberInput input {
        background-color: #1a1a1a !important;
        color: #00f0ff !important;
        border: 1px solid #00f0ff !important;
        border-radius: 5px !important;
    }
    
    .stNumberInput label {
        color: #ffffff !important;
    }
    
    /* Botones */
    .stButton button {
        background: linear-gradient(135deg, #00f0ff 0%, #0088cc 100%) !important;
        color: #0a0a0a !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
        padding: 10px 20px !important;
    }
    
    .stButton button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 40px rgba(0, 240, 255, 0.4) !important;
        background: linear-gradient(135deg, #00ffff 0%, #0099dd 100%) !important;
    }
    
    /* Pestañas en cian neón */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #00f0ff !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 3px solid #00f0ff !important;
        background-color: rgba(0, 240, 255, 0.05) !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        background-color: transparent !important;
    }
    
    /* Ecuaciones - caja con borde cian */
    .equation-box {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        border: 2px solid #00f0ff !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin: 20px 0 !important;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.1) !important;
    }
    
    .equation-box h4 {
        color: #00f0ff !important;
        margin-top: 15px !important;
    }
    
    .equation-box p {
        color: #cccccc !important;
    }
    
    /* Caja de Taylor - texto amarillo neón */
    .taylor-box {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        border: 2px solid #00f0ff !important;
        border-radius: 10px !important;
        padding: 30px !important;
        margin: 20px 0 !important;
        text-align: center !important;
        box-shadow: 0 0 40px rgba(0, 240, 255, 0.15) !important;
    }
    
    .taylor-box p {
        color: #ffff00 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        text-shadow: 0 0 30px rgba(255, 255, 0, 0.3) !important;
    }
    
    /* Tablas en modo oscuro */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border-collapse: collapse !important;
    }
    
    .dataframe thead tr th {
        background-color: #00f0ff !important;
        color: #0a0a0a !important;
        font-weight: 700 !important;
        padding: 10px !important;
    }
    
    .dataframe tbody tr td {
        border: 1px solid #333333 !important;
        color: #ffffff !important;
        padding: 8px !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: rgba(0, 240, 255, 0.05) !important;
    }
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    ::-webkit-scrollbar-thumb {
        background: #00f0ff;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00d0e0;
    }
    
    /* Selectbox en modo oscuro */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #ffffff !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
        border: 1px solid #00f0ff !important;
        border-radius: 5px !important;
    }
    
    /* Info, Warning, Success, Error boxes */
    .stAlert {
        background-color: #1a1a1a !important;
        border-left: 4px solid #00f0ff !important;
    }
    
    .stAlert div {
        color: #ffffff !important;
    }
    
    /* Captions y textos pequeños */
    .stCaption, .caption {
        color: #888888 !important;
    }
    
    /* Métricas custom (para velocidad promedio) */
    .custom-metric {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%) !important;
        border: 1px solid #00f0ff !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.05) !important;
    }
    
    .custom-metric .label {
        color: #b0b0b0 !important;
        font-size: 14px !important;
        font-weight: 400 !important;
    }
    
    .custom-metric .value {
        color: #00f0ff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        text-shadow: 0 0 30px rgba(0, 240, 255, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Funciones matemáticas ---
def velocity_profile(r, R, delta, rho, mu, g):
    """Perfil de velocidad exacto: vz(r)"""
    if r <= R:
        return 0.0
    a = (R + delta) / R
    return (rho * g * R**2 / (4 * mu)) * (1 - (r/R)**2 + 2 * a**2 * np.log(r/R))

def mass_flow_exact(R, delta, rho, mu, g):
    """Flujo másico exacto"""
    if R <= 0 or delta <= 0:
        return 0.0
    a = (R + delta) / R
    a2 = a**2
    a4 = a**4
    return (np.pi * rho**2 * g * R**4 / (8 * mu)) * (4 * a4 * np.log(a) - (3 * a4 - 4 * a2 + 1))

def average_velocity_exact(R, delta, rho, mu, g):
    """Velocidad promedio exacta"""
    if R <= 0 or delta <= 0:
        return 0.0
    a = (R + delta) / R
    a2 = a**2
    a4 = a**4
    return (rho * g * R**2 / (8 * mu)) * ((4 * a4 * np.log(a) / (a2 - 1)) - (3 * a2 - 1))

def mass_flow_taylor(R, delta, rho, mu, g):
    """Flujo másico aproximado (Taylor)"""
    if R <= 0 or delta <= 0:
        return 0.0
    return (2 * np.pi * R * rho**2 * g * delta**3) / (3 * mu)

def max_velocity(R, delta, rho, mu, g):
    """Velocidad máxima (en r = R + delta)"""
    if R <= 0 or delta <= 0:
        return 0.0
    a = (R + delta) / R
    return (rho * g * R**2 / (4 * mu)) * (1 - a**2 + 2 * a**2 * np.log(a))

def find_critical_delta(R, rho, mu, g, delta_range):
    """Encuentra el delta donde el error es 5%"""
    errors = []
    deltas = np.linspace(delta_range[0], delta_range[1], 1000)
    for d in deltas:
        if d <= 0:
            continue
        m_exact = mass_flow_exact(R, d, rho, mu, g)
        m_taylor = mass_flow_taylor(R, d, rho, mu, g)
        if m_exact > 0:
            error = abs(m_exact - m_taylor) / m_exact * 100
            errors.append(error)
        else:
            errors.append(100.0)
    
    # Interpolación para encontrar delta al 5%
    valid_indices = [i for i, e in enumerate(errors) if e > 5]
    if valid_indices and errors[0] > 5:
        for i in range(len(errors) - 1):
            if errors[i] >= 5 and errors[i+1] <= 5:
                delta_i = deltas[i]
                delta_j = deltas[i+1]
                error_i = errors[i]
                error_j = errors[i+1]
                if error_i != error_j:
                    delta_5 = delta_i + (5 - error_i) * (delta_j - delta_i) / (error_j - error_i)
                    return delta_5
    return None

# --- Función para crear visualización 3D ---
def create_3d_visualization(R, delta):
    """Crea la visualización 3D con Plotly"""
    # Crear el cilindro interior (tubo sólido)
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, 1, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_tube = R * np.cos(theta_grid)
    y_tube = R * np.sin(theta_grid)
    z_tube = z_grid
    
    # Crear la película líquida (capa exterior)
    R_ext = R + delta
    theta_film = np.linspace(0, 2*np.pi, 50)
    z_film = np.linspace(0, 1, 30)
    theta_film_grid, z_film_grid = np.meshgrid(theta_film, z_film)
    x_film = R_ext * np.cos(theta_film_grid)
    y_film = R_ext * np.sin(theta_film_grid)
    z_film_grid = z_film_grid
    
    # Flechas de velocidad (conos)
    n_arrows = 4
    arrow_positions = []
    for i in range(n_arrows):
        theta_arrow = 2 * np.pi * i / n_arrows
        r_arrow = R + delta * 0.5
        x_arrow = r_arrow * np.cos(theta_arrow)
        y_arrow = r_arrow * np.sin(theta_arrow)
        z_arrow = 0.3
        arrow_positions.append((x_arrow, y_arrow, z_arrow))
    
    # Crear figura
    fig = go.Figure()
    
    # Cilindro interno (tubo) - gris oscuro con gradiente
    fig.add_trace(go.Surface(
        x=x_tube, y=y_tube, z=z_tube,
        colorscale=[[0, '#333333'], [0.5, '#555555'], [1, '#777777']],
        opacity=1.0,
        showscale=False,
        name='Tubo interno',
        hovertemplate='<b>Tubo interno</b><br>Radio: %{x:.3f}<extra></extra>'
    ))
    
    # Película líquida - translúcida cian
    fig.add_trace(go.Surface(
        x=x_film, y=y_film, z=z_film_grid,
        colorscale=[[0, 'rgba(0, 240, 255, 0.2)'], [0.5, 'rgba(0, 200, 255, 0.3)'], [1, 'rgba(0, 150, 255, 0.4)']],
        opacity=0.4,
        showscale=False,
        name='Película líquida',
        hovertemplate='<b>Película líquida</b><br>Radio: %{x:.3f}<extra></extra>'
    ))
    
    # Flechas de velocidad (conos) - verde neón
    for x, y, z in arrow_positions:
        fig.add_trace(go.Cone(
            x=[x], y=[y], z=[z],
            u=[0], v=[0], w=[-0.3],
            colorscale=[[0, '#00ff00'], [1, '#00ff88']],
            showscale=False,
            name='Dirección del flujo',
            sizemode='absolute',
            sizeref=0.2,
            hovertemplate='<b>Flujo descendente</b><extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', color='#ffffff', gridcolor='#333333', backgroundcolor='#0a0a0a'),
            yaxis=dict(title='Y', color='#ffffff', gridcolor='#333333', backgroundcolor='#0a0a0a'),
            zaxis=dict(title='Z', color='#ffffff', gridcolor='#333333', backgroundcolor='#0a0a0a'),
            bgcolor='#0a0a0a',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.5),
                up=dict(x=0, y=0, z=1)
            )
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#ffffff', family='Arial, sans-serif'),
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        height=500,
    )
    
    return fig

# --- Función para graficar perfil de velocidad ---
def create_velocity_profile(R, delta, rho, mu, g):
    """Crea el gráfico del perfil de velocidad radial"""
    r_vals = np.linspace(R, R + delta, 200)
    vz_vals = [velocity_profile(r, R, delta, rho, mu, g) for r in r_vals]
    
    fig = go.Figure()
    
    # Perfil con relleno cian translúcido
    fig.add_trace(go.Scatter(
        x=r_vals * 1000,
        y=vz_vals,
        fill='tozeroy',
        fillcolor='rgba(0, 240, 255, 0.15)',
        line=dict(color='#00f0ff', width=3),
        name='v_z(r)',
        hovertemplate='<b>r = %{x:.2f} mm</b><br>v_z = %{y:.4f} m/s<extra></extra>'
    ))
    
    # Línea vertical para R (pared del tubo)
    fig.add_vline(
        x=R*1000, 
        line_dash="dash", 
        line_color="#ffffff",
        annotation_text=f"<b>R = {R*1000:.1f} mm</b>",
        annotation_position="top",
        annotation_font=dict(color='#ffffff', size=12)
    )
    
    # Línea vertical para R+delta (superficie libre)
    fig.add_vline(
        x=(R+delta)*1000, 
        line_dash="dash", 
        line_color="#00f0ff",
        annotation_text=f"<b>R+δ = {(R+delta)*1000:.1f} mm</b>",
        annotation_position="bottom",
        annotation_font=dict(color='#00f0ff', size=12)
    )
    
    # Añadir punto de velocidad máxima
    v_max = max_velocity(R, delta, rho, mu, g)
    fig.add_trace(go.Scatter(
        x=[(R+delta)*1000],
        y=[v_max],
        mode='markers',
        marker=dict(color='#ffff00', size=12, symbol='star', line=dict(color='#ffffff', width=2)),
        name='Vz máx',
        hovertemplate='<b>Vz máx = %{y:.4f} m/s</b><extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Perfil de Velocidad Radial Vz(r)</b>',
            font=dict(color='#ffffff', size=20),
            x=0.5
        ),
        xaxis=dict(
            title='<b>r [mm]</b>',
            color='#ffffff',
            gridcolor='#333333',
            gridwidth=1,
            zerolinecolor='#444444',
            title_font=dict(size=14)
        ),
        yaxis=dict(
            title='<b>v_z [m/s]</b>',
            color='#ffffff',
            gridcolor='#333333',
            gridwidth=1,
            zerolinecolor='#444444',
            title_font=dict(size=14)
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#ffffff', family='Arial, sans-serif'),
        showlegend=True,
        legend=dict(
            font=dict(color='#ffffff'),
            bgcolor='rgba(10, 10, 10, 0.8)',
            bordercolor='#00f0ff',
            borderwidth=1
        ),
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        hovermode='x'
    )
    
    return fig

# --- Función para crear gráficas de análisis de Taylor ---
def create_taylor_plots(R, rho, mu, g, current_delta):
    """Crea las gráficas de comparación para el análisis de Taylor"""
    delta_min = 0.0001
    delta_max = max(current_delta * 3, 0.005)
    
    deltas = np.linspace(delta_min, delta_max, 500)
    m_exact_list = []
    m_taylor_list = []
    errors = []
    
    for d in deltas:
        m_exact = mass_flow_exact(R, d, rho, mu, g)
        m_taylor = mass_flow_taylor(R, d, rho, mu, g)
        m_exact_list.append(m_exact)
        m_taylor_list.append(m_taylor)
        if m_exact > 0:
            error = abs(m_exact - m_taylor) / m_exact * 100
            errors.append(error)
        else:
            errors.append(100)
    
    # Gráfica A: Comparación de flujos másicos
    fig1 = go.Figure()
    
    # Línea exacta - cian neón
    fig1.add_trace(go.Scatter(
        x=deltas*1000, y=m_exact_list,
        mode='lines',
        name='<b>ṁ exacto</b>',
        line=dict(color='#00f0ff', width=3),
        hovertemplate='<b>δ = %{x:.3f} mm</b><br>ṁ exacto = %{y:.6e} kg/s<extra></extra>'
    ))
    
    # Línea Taylor - rojo neón
    fig1.add_trace(go.Scatter(
        x=deltas*1000, y=m_taylor_list,
        mode='lines',
        name='<b>ṁ Taylor</b>',
        line=dict(color='#ff6b6b', width=3, dash='dash'),
        hovertemplate='<b>δ = %{x:.3f} mm</b><br>ṁ Taylor = %{y:.6e} kg/s<extra></extra>'
    ))
    
    # Marcador del punto actual - amarillo neón
    m_exact_current = mass_flow_exact(R, current_delta, rho, mu, g)
    m_taylor_current = mass_flow_taylor(R, current_delta, rho, mu, g)
    fig1.add_trace(go.Scatter(
        x=[current_delta*1000],
        y=[m_exact_current],
        mode='markers',
        name='<b>Operación actual</b>',
        marker=dict(
            color='#ffff00', 
            size=15, 
            symbol='circle', 
            line=dict(color='#ffffff', width=3)
        ),
        hovertemplate='<b>δ = %{x:.3f} mm</b><br>ṁ = %{y:.6e} kg/s<extra></extra>'
    ))
    
    fig1.update_layout(
        title=dict(
            text='<b>Flujo Másico vs Espesor de Película</b>',
            font=dict(color='#ffffff', size=18),
            x=0.5
        ),
        xaxis=dict(
            title='<b>δ [mm]</b>',
            color='#ffffff',
            gridcolor='#333333',
            title_font=dict(size=14)
        ),
        yaxis=dict(
            title='<b>ṁ [kg/s]</b>',
            color='#ffffff',
            gridcolor='#333333',
            title_font=dict(size=14)
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#ffffff', family='Arial, sans-serif'),
        legend=dict(
            font=dict(color='#ffffff'),
            bgcolor='rgba(10, 10, 10, 0.8)',
            bordercolor='#00f0ff',
            borderwidth=1
        ),
        height=350,
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode='x'
    )
    
    # Gráfica B: Error relativo
    fig2 = go.Figure()
    
    # Error con relleno rosa neón
    fig2.add_trace(go.Scatter(
        x=deltas*1000, y=errors,
        mode='lines',
        name='<b>Error relativo</b>',
        fill='tozeroy',
        fillcolor='rgba(255, 105, 180, 0.2)',
        line=dict(color='#ff69b4', width=3),
        hovertemplate='<b>δ = %{x:.3f} mm</b><br>Error = %{y:.2f} %<extra></extra>'
    ))
    
    # Línea de 5% - amarilla
    fig2.add_hline(
        y=5, 
        line_dash="dash", 
        line_color="#ffff00",
        annotation_text="<b>5%</b>",
        annotation_position="top right",
        annotation_font=dict(color='#ffff00', size=14)
    )
    
    # Encontrar punto crítico donde error = 5%
    delta_5 = find_critical_delta(R, rho, mu, g, (delta_min, delta_max))
    if delta_5 is not None and delta_5 > 0:
        error_5 = 5.0
        fig2.add_trace(go.Scatter(
            x=[delta_5*1000],
            y=[error_5],
            mode='markers+text',
            name=f'<b>δ = {delta_5*1000:.2f} mm</b>',
            marker=dict(color='#ffff00', size=14, symbol='star', line=dict(color='#ffffff', width=2)),
            text=[f'<b>δ = {delta_5*1000:.2f} mm</b>'],
            textposition='top center',
            textfont=dict(color='#ffff00', size=12)
        ))
    
    fig2.update_layout(
        title=dict(
            text='<b>Error Relativo vs Espesor de Película</b>',
            font=dict(color='#ffffff', size=18),
            x=0.5
        ),
        xaxis=dict(
            title='<b>δ [mm]</b>',
            color='#ffffff',
            gridcolor='#333333',
            title_font=dict(size=14)
        ),
        yaxis=dict(
            title='<b>Error [%]</b>',
            color='#ffffff',
            gridcolor='#333333',
            title_font=dict(size=14)
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#ffffff', family='Arial, sans-serif'),
        legend=dict(
            font=dict(color='#ffffff'),
            bgcolor='rgba(10, 10, 10, 0.8)',
            bordercolor='#00f0ff',
            borderwidth=1
        ),
        height=350,
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode='x'
    )
    
    return fig1, fig2

# --- Función para generar tabla de datos de Taylor ---
def generate_taylor_table(R, rho, mu, g, current_delta):
    """Genera la tabla de comparación para el análisis de Taylor"""
    deltas_compare = [0.0001, 0.0010, 0.0050, 0.0100, 0.0200, 0.0300]
    if current_delta not in deltas_compare:
        deltas_compare.append(current_delta)
    deltas_compare.sort()
    
    data = []
    for d in deltas_compare:
        m_exact = mass_flow_exact(R, d, rho, mu, g)
        m_taylor = mass_flow_taylor(R, d, rho, mu, g)
        if m_exact > 0:
            error = abs(m_exact - m_taylor) / m_exact * 100
        else:
            error = 100
        is_current = (abs(d - current_delta) < 1e-8)
        data.append({
            'δ [mm]': d*1000,
            'ṁ exacto [kg/s]': m_exact,
            'ṁ Taylor [kg/s]': m_taylor,
            'Error [%]': error,
            'Actual': '✅' if is_current else ''
        })
    
    df = pd.DataFrame(data)
    # Formatear números
    df['ṁ exacto [kg/s]'] = df['ṁ exacto [kg/s]'].apply(lambda x: f"{x:.6e}")
    df['ṁ Taylor [kg/s]'] = df['ṁ Taylor [kg/s]'].apply(lambda x: f"{x:.6e}")
    df['Error [%]'] = df['Error [%]'].apply(lambda x: f"{x:.2f}")
    df['δ [mm]'] = df['δ [mm]'].apply(lambda x: f"{x:.4f}")
    
    return df

# --- Sidebar de parámetros ---
with st.sidebar:
    st.markdown("## ⚙️ Parámetros")
    st.markdown("---")
    
    # Sliders
    rho = st.slider(
        "Densidad ρ [kg/m³]", 
        min_value=800, 
        max_value=1500, 
        value=1000, 
        step=10,
        help="Densidad del fluido"
    )
    
    mu = st.slider(
        "Viscosidad μ [Pa·s]", 
        min_value=0.01, 
        max_value=1.00, 
        value=0.10, 
        step=0.01, 
        format="%.2f",
        help="Viscosidad dinámica del fluido"
    )
    
    # Number inputs para geometría
    R = st.number_input(
        "Radio del tubo R [m]", 
        min_value=0.010, 
        max_value=0.500, 
        value=0.030, 
        step=0.001, 
        format="%.3f",
        help="Radio externo del tubo"
    )
    
    delta = st.number_input(
        "Espesor de película δ [m]", 
        min_value=0.001, 
        max_value=0.050, 
        value=0.003, 
        step=0.001, 
        format="%.3f",
        help="Espesor de la película líquida"
    )
    
    # Botón de reset
    if st.button("🔄 Resetear valores", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Estado del sistema")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("a = (R+δ)/R", f"{((R+delta)/R):.4f}")
    with col2:
        st.metric("δ/R", f"{(delta/R):.4f}")
    
    if delta/R < 0.1:
        st.success("✅ Aprox. película delgada válida (δ/R < 0.1)")
    else:
        st.warning("⚠️ Aprox. película delgada no válida (δ/R > 0.1)")

# --- Cálculos principales ---
g = 9.81  # Aceleración de gravedad

# Calcular métricas
m_exact = mass_flow_exact(R, delta, rho, mu, g)
m_taylor = mass_flow_taylor(R, delta, rho, mu, g)
v_max = max_velocity(R, delta, rho, mu, g)
v_avg = average_velocity_exact(R, delta, rho, mu, g)

if m_exact > 0:
    error = abs(m_exact - m_taylor) / m_exact * 100
else:
    error = 0

# --- Cabecera principal con métricas ---
st.markdown("## 🌊 Flujo en Película Cilíndrica Descendente")
st.markdown("### Bird, Stewart & Lightfoot - Problema 2B.6")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Flujo Másico Exacto",
        value=f"{m_exact:.6f} kg/s",
        delta=None
    )

with col2:
    st.metric(
        label="Velocidad Máxima Vz",
        value=f"{v_max:.4f} m/s",
        delta=None
    )

with col3:
    st.metric(
        label="Flujo Másico Taylor",
        value=f"{m_taylor:.6f} kg/s",
        delta=None
    )

with col4:
    color = "#ff6b6b" if error > 5 else "#00f0ff"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
                border: 1px solid {color};
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 0 30px rgba(0, 240, 
