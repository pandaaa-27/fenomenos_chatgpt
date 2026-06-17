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

# --- Inyección de CSS para modo oscuro premium ---
st.markdown("""
<style>
    /* Fondo general oscuro */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* Sidebar oscuro */
    .css-1d391kg {
        background-color: #141414 !important;
    }
    
    /* Títulos en blanco puro */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Texto de métricas en blanco puro */
    .stMetric {
        color: #ffffff !important;
    }
    
    /* Valores de métricas en cian neón */
    .stMetric .css-1xarl3l {
        color: #00f0ff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(0, 240, 255, 0.3) !important;
    }
    
    /* Labels de métricas en gris claro */
    .stMetric .css-1d391kg {
        color: #b0b0b0 !important;
    }
    
    /* Sliders con estilo cian */
    .stSlider .css-1c36l6u {
        color: #00f0ff !important;
    }
    
    /* Inputs numéricos */
    .stNumberInput input {
        background-color: #1a1a1a !important;
        color: #00f0ff !important;
        border: 1px solid #00f0ff !important;
    }
    
    /* Botones */
    .stButton button {
        background-color: #00f0ff !important;
        color: #0a0a0a !important;
        font-weight: 700 !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        background-color: #00d0e0 !important;
        transform: scale(1.05) !important;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.4) !important;
    }
    
    /* Pestañas en cian neón */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #00f0ff !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 3px solid #00f0ff !important;
    }
    
    /* Sección de ecuaciones con cian */
    .equation-box {
        background-color: #0a0a0a !important;
        border: 2px solid #00f0ff !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin: 20px 0 !important;
        box-shadow: 0 0 30px rgba(0, 240, 255, 0.1) !important;
    }
    
    /* Caja de Taylor con texto amarillo neón */
    .taylor-box {
        background-color: #0a0a0a !important;
        border: 2px solid #00f0ff !important;
        border-radius: 10px !important;
        padding: 25px !important;
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
    
    /* Tablas en modo oscuro */
    .dataframe {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border-collapse: collapse !important;
    }
    
    .dataframe th {
        background-color: #00f0ff !important;
        color: #0a0a0a !important;
        font-weight: 700 !important;
    }
    
    .dataframe td {
        border: 1px solid #333333 !important;
        color: #ffffff !important;
    }
    
    /* Estilo para métricas dentro de la página */
    .metric-container {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
        border: 1px solid #00f0ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.05);
    }
    
    .metric-label {
        color: #b0b0b0 !important;
        font-size: 14px !important;
        font-weight: 400 !important;
    }
    
    .metric-value {
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
        # Buscar el punto donde cruza 5%
        for i in range(len(errors) - 1):
            if errors[i] >= 5 and errors[i+1] <= 5:
                # Interpolación lineal
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
    # Posiciones de los conos en la película
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
    
    # Cilindro interno (tubo) - opaco
    fig.add_trace(go.Surface(
        x=x_tube, y=y_tube, z=z_tube,
        colorscale=[[0, '#444444'], [1, '#666666']],
        opacity=1.0,
        showscale=False,
        name='Tubo interno'
    ))
    
    # Película líquida - translúcida cian
    fig.add_trace(go.Surface(
        x=x_film, y=y_film, z=z_film_grid,
        colorscale=[[0, '#00f0ff'], [1, '#0088ff']],
        opacity=0.3,
        showscale=False,
        name='Película líquida'
    ))
    
    # Flechas de velocidad (conos)
    for x, y, z in arrow_positions:
        fig.add_trace(go.Cone(
            x=[x], y=[y], z=[z],
            u=[0], v=[0], w=[-0.3],
            colorscale=[[0, '#00ff00'], [1, '#00ff00']],
            showscale=False,
            name='Dirección del flujo',
            sizemode='absolute',
            sizeref=0.2,
        ))
    
    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', color='#ffffff', gridcolor='#333333'),
            yaxis=dict(title='Y', color='#ffffff', gridcolor='#333333'),
            zaxis=dict(title='Z', color='#ffffff', gridcolor='#333333'),
            bgcolor='#0a0a0a',
            aspectmode='data',
        ),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#ffffff'),
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
    
    # Perfil con relleno
    fig.add_trace(go.Scatter(
        x=r_vals * 1000,  # Convertir a mm
        y=vz_vals,
        fill='tozeroy',
        fillcolor='rgba(0, 240, 255, 0.2)',
        line=dict(color='#00f0ff', width=3),
        name='v_z(r)'
    ))
    
    # Línea vertical para R
    fig.add_vline(x=R*1000, line_dash="dash", line_color="#ffffff", 
                  annotation_text=f"R = {R*1000:.1f} mm", 
                  annotation_position="top")
    
    # Línea vertical para R+delta
    fig.add_vline(x=(R+delta)*1000, line_dash="dash", line_color="#00f0ff",
                  annotation_text=f"R+δ = {(R+delta)*1000:.1f} mm",
                  annotation_position="bottom")
    
    fig.update_layout(
        title=dict(text='Perfil de Velocidad Radial Vz(r)', font=dict(color='#ffffff', size=20)),
        xaxis=dict(title='r [mm]', color='#ffffff', gridcolor='#333333'),
        yaxis=dict(title='v_z [m/s]', color='#ffffff', gridcolor='#333333'),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#ffffff'),
        showlegend=True,
        legend=dict(font=dict(color='#ffffff')),
        height=400,
        margin=dict(l=60, r=40, t=50, b=60),
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
    fig1.add_trace(go.Scatter(
        x=deltas*1000, y=m_exact_list,
        mode='lines',
        name='ṁ exacto',
        line=dict(color='#00f0ff', width=3)
    ))
    fig1.add_trace(go.Scatter(
        x=deltas*1000, y=m_taylor_list,
        mode='lines',
        name='ṁ Taylor',
        line=dict(color='#ff6b6b', width=3, dash='dash')
    ))
    # Marcador del punto actual
    m_exact_current = mass_flow_exact(R, current_delta, rho, mu, g)
    m_taylor_current = mass_flow_taylor(R, current_delta, rho, mu, g)
    fig1.add_trace(go.Scatter(
        x=[current_delta*1000],
        y=[m_exact_current],
        mode='markers',
        name='Operación actual',
        marker=dict(color='#ffff00', size=15, symbol='circle', line=dict(color='#ffffff', width=2))
    ))
    
    fig1.update_layout(
        title=dict(text='Flujo Másico vs Espesor de Película', font=dict(color='#ffffff', size=18)),
        xaxis=dict(title='δ [mm]', color='#ffffff', gridcolor='#333333'),
        yaxis=dict(title='ṁ [kg/s]', color='#ffffff', gridcolor='#333333'),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#ffffff'),
        legend=dict(font=dict(color='#ffffff')),
        height=350,
        margin=dict(l=60, r=40, t=50, b=60),
    )
    
    # Gráfica B: Error relativo
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=deltas*1000, y=errors,
        mode='lines',
        name='Error relativo',
        fill='tozeroy',
        fillcolor='rgba(255, 105, 180, 0.3)',
        line=dict(color='#ff69b4', width=3)
    ))
    # Línea de 5%
    fig2.add_hline(y=5, line_dash="dash", line_color="#ffff00",
                   annotation_text="5%", annotation_position="top right")
    
    # Encontrar punto crítico donde error = 5%
    delta_5 = find_critical_delta(R, rho, mu, g, (delta_min, delta_max))
    if delta_5 is not None and delta_5 > 0:
        m_exact_5 = mass_flow_exact(R, delta_5, rho, mu, g)
        error_5 = 5.0
        fig2.add_trace(go.Scatter(
            x=[delta_5*1000],
            y=[error_5],
            mode='markers+text',
            name=f'δ = {delta_5*1000:.2f} mm',
            marker=dict(color='#ffff00', size=12, symbol='star'),
            text=[f"<b>δ = {delta_5*1000:.2f} mm</b>"],
            textposition="top center"
        ))
    
    fig2.update_layout(
        title=dict(text='Error Relativo vs Espesor de Película', font=dict(color='#ffffff', size=18)),
        xaxis=dict(title='δ [mm]', color='#ffffff', gridcolor='#333333'),
        yaxis=dict(title='Error [%]', color='#ffffff', gridcolor='#333333'),
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        font=dict(color='#ffffff'),
        legend=dict(font=dict(color='#ffffff')),
        height=350,
        margin=dict(l=60, r=40, t=50, b=60),
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
    
    # Inicializar valores en session_state si no existen
    if 'reset' not in st.session_state:
        st.session_state.reset = False
    
    # Sliders
    rho = st.slider("Densidad ρ [kg/m³]", min_value=800, max_value=1500, value=1000, step=10)
    mu = st.slider("Viscosidad μ [Pa·s]", min_value=0.01, max_value=1.00, value=0.10, step=0.01, format="%.2f")
    
    # Number inputs para geometría
    R = st.number_input("Radio del tubo R [m]", min_value=0.010, max_value=0.500, value=0.030, step=0.001, format="%.3f")
    delta = st.number_input("Espesor de película δ [m]", min_value=0.001, max_value=0.050, value=0.003, step=0.001, format="%.3f")
    
    # Botón de reset
    if st.button("🔄 Resetear valores", use_container_width=True):
        st.session_state.reset = True
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Estado del sistema")
    st.markdown(f"**a = (R+δ)/R =** {((R+delta)/R):.4f}")
    st.markdown(f"**δ/R =** {(delta/R):.4f}")
    
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
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Flujo Másico Exacto</div>
        <div class="metric-value">{:.6f} kg/s</div>
    </div>
    """.format(m_exact), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Velocidad Máxima Vz</div>
        <div class="metric-value">{:.4f} m/s</div>
    </div>
    """.format(v_max), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Flujo Másico Taylor</div>
        <div class="metric-value">{:.6f} kg/s</div>
    </div>
    """.format(m_taylor), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-container">
        <div class="metric-label">Error Relativo</div>
        <div class="metric-value" style="color: {}">{:.2f} %</div>
    </div>
    """.format('#ff6b6b' if error > 5 else '#00f0ff', error), unsafe_allow_html=True)

st.markdown("---")

# --- Pestañas ---
tab1, tab2 = st.tabs(["📊 Visualización Física", "📈 Análisis de Taylor"])

# --- Tab 1: Visualización Física ---
with tab1:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 🌐 Representación 3D del Sistema")
        fig_3d = create_3d_visualization(R, delta)
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption("Cilindro interior: tubo sólido | Capa exterior: película líquida (cian translúcido) | Flechas verdes: dirección del flujo")
    
    with col_right:
        st.markdown("### 📈 Perfil de Velocidad Radial")
        fig_vz = create_velocity_profile(R, delta, rho, mu, g)
        st.plotly_chart(fig_vz, use_container_width=True)
        
        # Métrica de velocidad promedio
        st.markdown("""
        <div class="metric-container" style="margin-top: 15px;">
            <div class="metric-label">Velocidad Promedio ⟨vz⟩</div>
            <div class="metric-value">{:.4f} m/s</div>
        </div>
        """.format(v_avg), unsafe_allow_html=True)
    
    # Ecuaciones
    st.markdown("---")
    st.markdown("### 📐 Ecuaciones del Modelo Cilíndrico")
    
    with st.container():
        st.markdown("""
        <div class="equation-box">
            <h4>Perfil de Velocidad Exacto</h4>
            $$v_z(r) = \\frac{\\rho g R^2}{4\\mu} \\left[1 - \\left(\\frac{r}{R}\\right)^2 + 2a^2 \\ln\\left(\\frac{r}{R}\\right)\\right]$$
            
            <h4>Flujo Másico Exacto</h4>
            $$\\dot{m} = \\frac{\\pi \\rho^2 g R^4}{8\\mu} \\left[4a^4 \\ln(a) - (3a^4 - 4a^2 + 1)\\right]$$
            
            <h4>Velocidad Promedio Exacta</h4>
            $$\\langle v_z \\rangle = \\frac{\\rho g R^2}{8\\mu} \\left[\\frac{4a^4 \\ln(a)}{a^2 - 1} - (3a^2 - 1)\\right]$$
            
            <p style="color: #b0b0b0; margin-top: 10px;">
                donde \(a = \\frac{R + \\delta}{R}\) y \(R \\leq r \\leq R + \\delta\)
            </p>
        </div>
        """, unsafe_allow_html=True)

# --- Tab 2: Análisis de Taylor ---
with tab2:
    # Caja de Taylor
    st.markdown("""
    <div class="taylor-box">
        <p>📐 Aproximación de Taylor (Película Delgada)</p>
        <p style="font-size: 32px; color: #ffff00 !important;">
            ṁ ≈ (2πRρ²gδ³) / (3μ)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_left2, col_right2 = st.columns([0.4, 0.6])
    
    with col_left2:
        st.markdown("### 📊 Tabla Comparativa")
        df_table = generate_taylor_table(R, rho, mu, g, delta)
        st.dataframe(df_table, use_container_width=True, height=400)
        st.caption("✅ = valor actual seleccionado en el slider")
    
    with col_right2:
        st.markdown("### 📈 Análisis de Sensibilidad")
        fig1, fig2 = create_taylor_plots(R, rho, mu, g, delta)
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666666; padding: 20px;">
    Desarrollado con ❤️ para el Curso de Fenómenos de Transporte<br>
    Basado en Bird, Stewart & Lightfoot - "Transport Phenomena" 2nd Edition
</div>
""", unsafe_allow_html=True)
