import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="WNBA Draft Fit Predictor",
    page_icon="🏀",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    predictions = pd.read_csv("data/processed/predictions_2026.csv")
    rookies = pd.read_csv("data/processed/rookies_clean.csv")
    validation = pd.read_csv("data/processed/validation_snapshot.csv")
    return predictions, rookies, validation

predictions, rookies, validation = load_data()

# Title
st.title("🏀 WNBA Draft Fit Predictor")
st.subheader("2026 Draft Class — Predicted Rookie Impact by Team Fit & NCAA Career Profile")
st.markdown("---")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Draft Class Size", "12 NCAA Players")
with col2:
    st.metric("Training Data", "86 Historical Rookies")
with col3:
    st.metric("Seasons Analyzed", "2019–2025")
with col4:
    st.metric("Model", "Ridge Regression (R² = 0.172)")

st.markdown("---")

# Predictions table
st.header("2026 Predicted Rookie Impact Scores")
display_df = predictions[['player', 'wnba_team', 'draft_pick', 'predicted_RIS']].sort_values('predicted_RIS', ascending=False).copy()
display_df.columns = ['Player', 'Team', 'Draft Pick', 'Predicted RIS']
display_df = display_df.reset_index(drop=True)
display_df.index += 1
st.dataframe(display_df, use_container_width=True)

st.markdown("---")

# Visualization section
st.header("Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Predicted RIS", "Draft Position vs Impact", "Feature Importance", "RIS Distribution"])

with tab1:
    st.image("visuals/predicted_ris_bar.png", use_container_width=True)

with tab2:
    st.image("visuals/draft_position_vs_ris.png", use_container_width=True)

with tab3:
    st.image("visuals/feature_importance.png", use_container_width=True)

with tab4:
    st.image("visuals/ris_distribution.png", use_container_width=True)

st.markdown("---")

# Player explorer
st.header("Player Explorer")
selected_player = st.selectbox("Select a player to explore", predictions['player'].sort_values())

player_data = predictions[predictions['player'] == selected_player].iloc[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Team", player_data['wnba_team'])
with col2:
    st.metric("Draft Pick", f"#{int(player_data['draft_pick'])}")
with col3:
    st.metric("Predicted RIS", f"{player_data['predicted_RIS']:.3f}")

st.markdown("---")
st.caption("Built by Charlize Andaya | Data: Sports Reference & Basketball Reference | Model: Ridge Regression")