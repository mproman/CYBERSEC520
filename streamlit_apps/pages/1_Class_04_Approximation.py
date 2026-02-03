import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Universal Approximation Theorem Demo")
st.markdown("""
This interactive visualization demonstrates how a neural network with a single hidden layer 
(combining multiple ReLU units) can approximate essentially any continuous function.
""")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")
n_segments = st.sidebar.slider(
    "Number of ReLU Segments (Neurons)", 
    min_value=2, 
    max_value=50, 
    value=5, 
    step=1,
    help="More segments = more neurons = better approximation"
)

# --- Logic (Replicating Class 4) ---
def relu(x):
    return np.maximum(0, x)

def target_function(x):
    """Complex target function with multiple frequencies"""
    return np.sin(x) + 0.5 * np.sin(3 * x)

# Generate data
x = np.linspace(-np.pi, np.pi, 1000)
y_true = target_function(x)

# Create knot points where ReLU "bends" occur
x_knots = np.linspace(-np.pi, np.pi, n_segments + 1)
y_knots = target_function(x_knots)

# Calculate slopes between knot points
slopes = np.diff(y_knots) / np.diff(x_knots)
intercepts = y_knots[:-1] - slopes * x_knots[:-1]

# Build piecewise linear approximation
y_approx = slopes[0] * x + intercepts[0]
for i in range(1, len(slopes)):
    delta_slope = slopes[i] - slopes[i-1]
    y_approx += delta_slope * relu(x - x_knots[i])

# Calculate approximation error
mse = np.mean((y_true - y_approx)**2)

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Main approximation plot
ax1.plot(x, y_true, 'b-', label='Target Function', linewidth=2.5, alpha=0.8)
ax1.plot(x, y_approx, 'r--', label='ReLU Approximation', linewidth=2.5)
ax1.plot(x_knots, y_knots, 'go', markersize=8, label='Knot Points', alpha=0.7)
ax1.set_title(f'Approximation with {n_segments} Segments', fontsize=14, fontweight='bold')
ax1.set_xlabel('Input (x)', fontsize=12)
ax1.set_ylabel('Output (y)', fontsize=12)
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'MSE: {mse:.4f}', transform=ax1.transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Error plot
error = np.abs(y_true - y_approx)
ax2.fill_between(x, error, alpha=0.5, color='red')
ax2.set_title('Approximation Error', fontsize=14, fontweight='bold')
ax2.set_xlabel('Input (x)', fontsize=12)
ax2.set_ylabel('Absolute Error', fontsize=12)
ax2.grid(True, alpha=0.3)

st.pyplot(fig)

st.caption(f"Mean Squared Error: {mse:.6f}")
