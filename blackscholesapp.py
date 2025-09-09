import streamlit as st
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go

st.set_page_config(page_title="Black-Scholes Model", layout="wide")
st.title("⚙️ Black-Scholes Call & Put Option Model")


def black_scholes(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

#sidebar inputs
st.sidebar.header("Option Parameters")
spot_price_single = st.sidebar.number_input(
    "Spot Price for Call/Put Value",
    min_value=0.0, max_value=500.0,
    value=100.0, step=1.0, format="%.2f"
)
K = st.sidebar.number_input(
    "Strike Price (K)",
    min_value=0.0, max_value=500.0,
    value=100.0, step=1.0, format="%.2f"
)
sigma = st.sidebar.number_input(
    "Volatility (σ)",
    min_value=0.0, max_value=1.0,
    value=0.2, step=0.01, format="%.4f"
)
r = st.sidebar.number_input(
    "Risk-Free Rate (r)",
    min_value=0.0, max_value=0.1,
    value=0.05, step=0.001, format="%.4f"
)
T = st.sidebar.number_input(
    "Time to Maturity (T, years)",
    min_value=0.0, max_value=10.0,
    value=1.0, step=0.25, format="%.4f"
)

#display single option vlaues

call_val = black_scholes(spot_price_single, K, T, r, sigma, "call")
put_val = black_scholes(spot_price_single, K, T, r, sigma, "put")

st.markdown("### Current Option Prices at Spot Price Input")
st.success(f"Call Option Value at Spot Price {spot_price_single:.2f}: **£{call_val:.2f}**")
st.error(f"Put Option Value at Spot Price {spot_price_single:.2f}: **£{put_val:.2f}**")
st.markdown("---")


#heatmap options

st.sidebar.markdown("---")
st.sidebar.header("Heatmap Options")

spot_min = st.sidebar.number_input(
    "Minimum Spot Price for Heatmaps",
    min_value=10.0, max_value=500.0,
    value=50.0, step=1.0, format="%.2f"
)
spot_max = st.sidebar.number_input(
    "Maximum Spot Price for Heatmaps",
    min_value=10.0, max_value=500.0,
    value=150.0, step=1.0, format="%.2f"
)

if spot_max <= spot_min:
    st.sidebar.error("Maximum Spot Price must be greater than Minimum Spot Price")

vol_min, vol_max = st.sidebar.slider(
    "Volatility Range (σ)",
    min_value=0.05, max_value=1.0,
    value=(0.1, 0.5),
    step=0.01,
    format="%.4f"
)
if vol_max <= vol_min:
    st.sidebar.error("Maximum Volatility must be greater than Minimum Volatility")

#prepare 10x10 grid
S_values = np.linspace(spot_min, spot_max, 10)
sigma_values = np.linspace(vol_min, vol_max, 10)

#calc option values for heatmap
call_matrix = np.array([
    [black_scholes(S, K, T, r, vol, "call") for S in S_values]
    for vol in sigma_values
])
put_matrix = np.array([
    [black_scholes(S, K, T, r, vol, "put") for S in S_values]
    for vol in sigma_values
])

def plot_heatmap(Z, x_vals, y_vals, title, colorscale):
    text = [[f"£{v:.2f}" for v in row] for row in Z]
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=x_vals,
        y=y_vals,
        text=text,
        texttemplate="%{text}",
        colorscale=colorscale,
        colorbar=dict(title="Option Value (£)")
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Spot Price (S)",
        yaxis_title="Volatility (σ)",
        height=500,
        dragmode=False,
        hovermode=False,
        xaxis=dict(
            fixedrange=True,
            tickmode='array',
            tickvals=x_vals,
            ticktext=[f"{v:.0f}" if v.is_integer() else f"{v:.2f}" for v in x_vals],
        ),
        yaxis=dict(
            fixedrange=True,
            tickmode='array',
            tickvals=y_vals,
            ticktext=[f"{v:.2f}" for v in y_vals],
            ticks="outside"
        ),
    )
    return fig

left, right = st.columns(2)
with left:
    st.plotly_chart(plot_heatmap(call_matrix, S_values, sigma_values, "Call Price Heatmap", "YlGnBu"), use_container_width=True)
with right:
    st.plotly_chart(plot_heatmap(put_matrix, S_values, sigma_values, "Put Price Heatmap", "YlOrRd"), use_container_width=True)
