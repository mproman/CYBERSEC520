import streamlit as st

st.set_page_config(
    page_title="Cybersecurity AI Demos",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Interactive Demos")
st.markdown("""
### CYBERSEC 520: Foundations of AI in Cybersecurity

Welcome to the interactive demonstration suite. Select a class module from the sidebar to begin exploring.

#### Available Modules:
- **Class 04:** Universal Approximation Theorem (ReLU Networks)
""")

st.sidebar.success("Select a demo above.")
