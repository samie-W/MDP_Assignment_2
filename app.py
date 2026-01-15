import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from mdpLogic import GridWorldMDP, value_iteration

st.set_page_config(page_title="MDP Visualizer", layout="wide")

st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: white;
        color: #1E1E1E;
    }
    
    /* Titles and Labels */
    h1, h2, h3, label, .stMarkdown {
        color: #1E1E1E !important;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }

    /* Selectbox/Dropdown styling */
    div[data-baseweb="select"] > div {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #D0D0D0;
    }

    /* Button styling */
    div.stButton > button {
        background-color: #1E1E1E;
        color: white;
        border-radius: 6px;
        width: 100%;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #404040;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("MDP Visualization Tool")
st.markdown("Analyze how optimal policies emerge through Value and Policy Iteration. [cite: 6]")
st.divider()


st.sidebar.header("Configuration")
algo = st.sidebar.selectbox(
    label="Choose Algorithm ", 
    options=["Value Iteration", "Policy Iteration"],
    help="Select the mathematical approach for optimization."
)
gamma = st.sidebar.slider(
    label="Discount Factor (γ) ", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.9,
    help="Determines the importance of future rewards. [cite: 17]"
)
run_btn = st.sidebar.button("Compute Optimal Strategy")


mdp = GridWorldMDP(gamma=gamma)

if run_btn:
   
    V, history = value_iteration(mdp)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("Value Heatmap ") 
 
        fig = px.imshow(V, text_auto=".2f", color_continuous_scale='Greens',
                        labels=dict(color="Utility"))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Learned Policy [cite: 24, 35]") 
        st.write("Numerical values representing state utilities after convergence: [cite: 36]")
  
        st.dataframe(pd.DataFrame(V).style.background_gradient(cmap='Greens'), use_container_width=True)

    st.success(f"Convergence achieved in {len(history)} iterations. [cite: 25, 36]")