import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import yfinance as yf

# === PAGE CONFIG ===
st.set_page_config(layout="wide", page_title="Quant-3D: Option Matrix")

# === CUSTOM CSS ===
st.markdown("""
<style>
    .stApp { background-color: #050b14; }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #0d1b2a;
        padding: 15px; border-radius: 8px;
        border: 1px solid #1b263b;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    div[data-testid="stMetricLabel"] { color: #8892b0 !important; font-size: 0.9rem; }
    div[data-testid="stMetricValue"] { color: #00f5d4 !important; font-family: 'monospace'; font-size: 1.8rem !important; }
    
    /* Inputs */
    .stNumberInput>div>div>input { color: #e6f1ff; background-color: #1b263b; }
    .stTextInput>div>div>input { color: #e6f1ff; background-color: #1b263b; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0d1b2a; border-right: 1px solid #1b263b; }
    
    /* Tabs */
    button[data-baseweb="tab"] { color: #8892b0; font-weight: bold; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #00f5d4; border-bottom: 2px solid #00f5d4; }
    
    h1 { color: #00f5d4; font-family: sans-serif; font-weight: 800; text-shadow: 0 0 10px rgba(0,245,212,0.3); }
    h3 { color: #8892b0; }
</style>
""", unsafe_allow_html=True)

# === BLACK-SCHOLES LOGIC ===
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 
    theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    
    return price, {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta}

# === SIDEBAR ===
st.sidebar.header("📡 Live Data Fetch")
ticker = st.sidebar.text_input("Enter Ticker (e.g. AAPL)", value="")

# Default Values
default_spot = 450.0
default_vol = 20.0

# Fetch Logic
if ticker:
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1y")
        
        if not history.empty:
            # Get Current Price
            current_price = history['Close'].iloc[-1]
            default_spot = float(current_price)
            
            # Calculate Historical Volatility (Annualized)
            returns = history['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100
            default_vol = float(volatility)
            
            st.sidebar.success(f"Loaded {ticker}: ${default_spot:.2f}, Vol: {default_vol:.1f}%")
        else:
            st.sidebar.error("Ticker not found.")
    except:
        st.sidebar.error("Error fetching data.")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Parameters")

# Inputs (Auto-filled if ticker found, otherwise defaults)
S = st.sidebar.number_input("Spot Price ($)", 0.0, 10000.0, default_spot, 1.0)
K = st.sidebar.number_input("Strike Price ($)", 0.0, 10000.0, default_spot * 1.05, 1.0) # Default Strike +5%
T_days = st.sidebar.slider("Days to Expiration", 1, 365, 30)
sigma = st.sidebar.slider("Implied Volatility (%)", 1.0, 200.0, default_vol, 0.5) / 100
r = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 20.0, 4.5, 0.1) / 100
opt_type = st.sidebar.radio("Option Type", ["Call", "Put"]).lower()

# Calculate
T = T_days / 365
price, greeks = black_scholes(S, K, T, r, sigma, opt_type)

# === MAIN UI ===
st.title("⚡ Quant-3D: Option Matrix")
if ticker:
    st.caption(f"Analyzing: **{ticker.upper()}** | Live Spot: **${S:.2f}** | Calc Volatility: **{default_vol:.1f}%**")
else:
    st.caption("Real-Time Black-Scholes Pricing Engine")

# METRICS
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Theoretical Price", f"${price:.2f}")
col2.metric("Delta", f"{greeks['Delta']:.3f}")
col3.metric("Gamma", f"{greeks['Gamma']:.3f}")
col4.metric("Vega", f"{greeks['Vega']:.3f}")
col5.metric("Theta", f"{greeks['Theta']:.3f}")

st.markdown("---")

# TABS
tab1, tab2 = st.tabs(["🧊 3D Volatility Surface", "📚 How It Works"])

with tab1:
    st.subheader("Price Sensitivity Surface")
    
    # Generate 3D Grid
    spot_range = np.linspace(S * 0.7, S * 1.3, 40)
    vol_range = np.linspace(0.1, 1.0, 40)
    X, Y = np.meshgrid(spot_range, vol_range)

    def get_z(spots, vols):
        D1 = (np.log(spots / K) + (r + 0.5 * vols ** 2) * T) / (vols * np.sqrt(T))
        D2 = D1 - vols * np.sqrt(T)
        if opt_type == 'call':
            return spots * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-D2) - spots * norm.cdf(-D1)

    Z = get_z(X, Y)

    # Plot
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    fig.update_layout(
        scene=dict(
            xaxis_title='Spot Price ($)',
            yaxis_title='Volatility',
            zaxis_title='Option Price ($)',
            bgcolor='#050b14'
        ),
        paper_bgcolor='#050b14',
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("The Black-Scholes Formula")
    st.markdown("""
    This app calculates the theoretical price of European-style options.
    
    ### How "Auto-Fetch" Works
    When you enter a ticker (e.g., **BPOP**), the app:
    1. Fetches the live closing price.
    2. Calculates the **Annualized Standard Deviation** of the last year's returns.
    3. Uses this as a proxy for **Implied Volatility ($\sigma$)**.
    
    ### The Equation
    """)
    st.latex(r'''
    C(S, t) = N(d_1)S - N(d_2)Ke^{-r(T-t)}
    ''')
