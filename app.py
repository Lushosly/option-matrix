import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

# === PAGE CONFIG ===
st.set_page_config(layout="wide", page_title="Quant-3D: Options Engine")

# === CSS ===
st.markdown("""
<style>
    .stApp { background-color: #050b14; color: #e0e1dd; }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #0d1b2a;
        padding: 15px; border-radius: 8px;
        border: 1px solid #1b263b;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] { color: #8892b0 !important; }
    div[data-testid="stMetricValue"] { color: #00f5d4 !important; font-family: 'monospace'; }
    
    /* Inputs */
    .stTextInput>div>div>input { color: #e6f1ff; background-color: #1b263b; }
    .stNumberInput>div>div>input { color: #e6f1ff; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1b2a; border-right: 1px solid #1b263b; }
    
    h1 { color: #00f5d4; font-family: sans-serif; font-weight: 800; }
    h3 { color: #8892b0; }
</style>
""", unsafe_allow_html=True)

# === BLACK-SCHOLES MATH ===
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    S: Spot Price
    K: Strike Price
    T: Time to Maturity (Years)
    r: Risk-free Interest Rate
    sigma: Volatility (IV)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    # Greeks
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 # Scaled
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    
    return price, {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

# === SIDEBAR CONTROLS ===
st.sidebar.header("🎛️ Parameters")

S = st.sidebar.number_input("Spot Price ($)", 1.0, 1000.0, 450.0, 1.0)
K = st.sidebar.number_input("Strike Price ($)", 1.0, 1000.0, 460.0, 1.0)
T_days = st.sidebar.slider("Days to Expiration", 1, 365, 30)
sigma = st.sidebar.slider("Implied Volatility (%)", 1.0, 200.0, 20.0, 0.5) / 100
r = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.5, 0.1) / 100
opt_type = st.sidebar.radio("Option Type", ["Call", "Put"]).lower()

# === CALCULATIONS ===
T = T_days / 365
price, greeks = black_scholes(S, K, T, r, sigma, opt_type)

# === MAIN UI ===
st.title("Quant-3D: Option Matrix")
st.markdown("Interactive **Black-Scholes Pricing Model** and **Volatility Surface** visualizer.")

# 1. HEADLINE METRICS
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Option Price", f"${price:.2f}")
col2.metric("Delta (Sensitivity)", f"{greeks['Delta']:.3f}")
col3.metric("Gamma (Acceleration)", f"{greeks['Gamma']:.3f}")
col4.metric("Vega (Volatility)", f"{greeks['Vega']:.3f}")
col5.metric("Theta (Time Decay)", f"{greeks['Theta']:.3f}")

# 2. 3D VISUALIZATION
st.markdown("---")
st.subheader("🧊 3D Price Surface (Spot Price vs. Volatility)")

# Generate 3D Grid
spot_range = np.linspace(S * 0.5, S * 1.5, 50)
vol_range = np.linspace(0.1, 1.0, 50)
X, Y = np.meshgrid(spot_range, vol_range)

# Vectorized BS Calculation for Grid
def get_surface_z(spots, vols):
    D1 = (np.log(spots / K) + (r + 0.5 * vols ** 2) * T) / (vols * np.sqrt(T))
    D2 = D1 - vols * np.sqrt(T)
    if opt_type == 'call':
        return spots * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-D2) - spots * norm.cdf(-D1)

Z = get_surface_z(X, Y)

# Plot 3D
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

fig.update_layout(
    title=f"{opt_type.title()} Price Sensitivity",
    scene=dict(
        xaxis_title='Spot Price ($)',
        yaxis_title='Volatility (IV)',
        zaxis_title='Option Price ($)',
        bgcolor='#050b14'
    ),
    paper_bgcolor='#050b14',
    font=dict(color='#e0e1dd'),
    margin=dict(l=0, r=0, b=0, t=30),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# 3. INTERACTIVE HEATMAP
with st.expander("📊 View Heatmap Data (P&L Simulation)"):
    # Create P/L Table based on Spot Price moves
    sim_spots = np.linspace(S * 0.8, S * 1.2, 10)
    sim_prices = []
    
    for s_sim in sim_spots:
        p_sim, _ = black_scholes(s_sim, K, T, r, sigma, opt_type)
        sim_prices.append({
            "Spot Price": s_sim,
            "Option Value": p_sim,
            "P/L ($)": (p_sim - price) * 100 # Assuming 1 contract (100 shares)
        })
    
    df_sim = pd.DataFrame(sim_prices)
    st.dataframe(df_sim.style.format({"Spot Price": "${:.2f}", "Option Value": "${:.2f}", "P/L ($)": "${:.2f}"}))

st.markdown("---")
st.markdown('<div style="font-size: 0.8rem; color: #555;">⚠️ DISCLAIMER: Educational tool. Uses Black-Scholes-Merton model assumptions. Not investment advice.</div>', unsafe_allow_html=True)
