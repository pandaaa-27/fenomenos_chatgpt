import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ==========================================================
# Título
# ==========================================================

st.set_page_config(layout="wide")

st.title("Flujo de una película descendente sobre un tubo")
st.subheader("Ejercicio 2B.6 - Bird, Stewart y Lightfoot")

st.markdown(
"""
Esta aplicación simula el flujo laminar de una película líquida
que desciende por el exterior de un tubo circular.

Se utiliza la solución analítica obtenida por Bird para:

- Perfil de velocidad
- Velocidad máxima
- Velocidad promedio
- Flujo volumétrico
- Flujo másico
"""
)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("Parámetros")

R = st.sidebar.slider(
    "Radio del tubo R (m)",
    0.005,
    0.20,
    0.05,
    0.001
)

delta = st.sidebar.slider(
    "Espesor de película δ (m)",
    0.0005,
    0.05,
    0.005,
    0.0005
)

rho = st.sidebar.slider(
    "Densidad ρ (kg/m³)",
    100.0,
    2000.0,
    1000.0,
    10.0
)

mu = st.sidebar.slider(
    "Viscosidad μ (Pa·s)",
    0.0005,
    5.0,
    0.01,
    0.0005
)

g = st.sidebar.slider(
    "Gravedad g (m/s²)",
    1.0,
    20.0,
    9.81,
    0.01
)

# ==========================================================
# Parámetros auxiliares
# ==========================================================

a = (R + delta)/R

# ==========================================================
# Perfil de velocidad
# ==========================================================

def velocity(r):

    return (
        rho*g*R**2/(4*mu)
        *
        (
            1
            - (r/R)**2
            + 2*a**2*np.log(r/R)
        )
    )

# Mallado radial

r = np.linspace(R, R + delta, 500)

vz = velocity(r)

# ==========================================================
# Flujo volumétrico
# ==========================================================

def integrand(r):

    return velocity(r)*r

Q = 2*np.pi*quad(
    integrand,
    R,
    R + delta
)[0]

# ==========================================================
# Área transversal de película
# ==========================================================

A = np.pi*((R+delta)**2 - R**2)

# ==========================================================
# Velocidad promedio
# ==========================================================

v_avg = Q/A

# ==========================================================
# Velocidad máxima
# ==========================================================

v_max = np.max(vz)

r_max = r[np.argmax(vz)]

# ==========================================================
# Flujo másico
# ==========================================================

m_dot = rho*Q

# ==========================================================
# Resultados
# ==========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Velocidad máxima",
    f"{v_max:.6f} m/s"
)

c2.metric(
    "Velocidad promedio",
    f"{v_avg:.6f} m/s"
)

c3.metric(
    "Flujo volumétrico",
    f"{Q:.6e} m³/s"
)

c4.metric(
    "Flujo másico",
    f"{m_dot:.6e} kg/s"
)

# ==========================================================
# Gráfico perfil de velocidad
# ==========================================================

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(
    r,
    vz,
    linewidth=3
)

ax.set_xlabel("r (m)")
ax.set_ylabel("vz (m/s)")
ax.set_title("Distribución de velocidad en la película")

ax.grid(True)

st.pyplot(fig)

# ==========================================================
# Información adicional
# ==========================================================

st.markdown("---")

st.write("### Información del máximo")

st.write(
f"""
La velocidad máxima ocurre aproximadamente en:

\[
r = {r_max:.5f}\;m
\]

\[
v_{{max}} = {v_max:.5f}\;m/s
\]
"""
)

# ==========================================================
# Perfil adimensional
# ==========================================================

st.write("### Perfil adimensional")

eta = (r-R)/delta

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.plot(
    eta,
    vz/v_max,
    linewidth=3
)

ax2.set_xlabel(r"$(r-R)/\delta$")
ax2.set_ylabel(r"$v/v_{max}$")
ax2.grid(True)

st.pyplot(fig2)

# ==========================================================
# Tabla de resultados
# ==========================================================

st.write("### Resumen")

st.dataframe(
{
    "Magnitud":[
        "Velocidad máxima",
        "Velocidad promedio",
        "Flujo volumétrico",
        "Flujo másico"
    ],
    "Valor":[
        v_max,
        v_avg,
        Q,
        m_dot
    ]
}
)
