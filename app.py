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

with st.expander("ℹ️ How to read this dashboard"):
    st.markdown("""
    **What is the Rookie Impact Score (RIS)?**
    
    RIS is a composite metric (0–1 scale) that predicts how much impact a rookie will have in their first WNBA season. It combines:
    - **Games played** — will they make the roster and stay healthy?
    - **Minutes per game** — will coaches trust them?
    - **Points, rebounds, assists** — will they produce?
    - **Win Shares per 40 min** — will they be efficient?
    
    **How were predictions made?**
    
    A Ridge Regression model was trained on 86 Round 1 rookies from 2019–2025, using their weighted NCAA career averages and draft position to predict WNBA rookie impact. Model performance was evaluated using both 5-fold cross validation (R² = 0.172) and leave-one-out cross validation (R² = 0.418). The true model performance likely falls between these estimates.
                
    **What does "above/below prediction" mean?**
    
    A positive delta (green) means the player is currently outperforming their predicted RIS. A negative delta (red) means they're underperforming. Note: with only 2–3 games played, these are very early signals.
    
    **Why are some players missing stats?**
    
    Cotie McMahon suffered a season-ending UCL tear. Taina Mair was waived before playing. These outcomes highlight a key model limitation — injury and roster decisions cannot be predicted from college stats alone.
    
    **Why are international players not included?**
    
    International prospects (e.g. Awa Fam, Nell Angloma) don't have NCAA stats, which are the foundation of this model. Their stats from European leagues aren't directly comparable to college basketball, so including them would require a separate translation model. They are noted as a future improvement.
    
    **Why only Round 1 picks?**
    
    Round 1 picks represent the highest-stakes decisions — the largest contracts, the most pressure, and the most scrutiny. They also have the most reliable data and the lowest rate of never playing at all. Round 2 and 3 picks are more likely to get waived before playing, which introduces noise into the model.""")
    

# Key metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Draft Class Size", "12 NCAA Players")
with col2:
    st.metric("Training Data", "86 Historical Rookies")
with col3:
    st.metric("Seasons Analyzed", "2019–2025")
with col4:
    st.metric("Model", "Ridge Regression")
    st.markdown("##### R² = 0.172 (5-fold CV) | R² = 0.418 (LOO CV)")

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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Predicted RIS", "Draft Position vs Impact", "Feature Importance", "RIS Distribution", "Historical Accuracy"])
with tab1:
    st.image("visuals/predicted_ris_bar.png", use_container_width=True)

with tab2:
    st.image("visuals/draft_position_vs_ris.png", use_container_width=True)

with tab3:
    st.image("visuals/feature_importance.png", use_container_width=True)

with tab4:
    st.image("visuals/ris_distribution.png", use_container_width=True)

with tab5:
    st.image("visuals/backtest_accuracy.png", use_container_width=True)
    
    # Load backtest results
    backtest = pd.read_csv("data/processed/backtest_results.csv")
    
    st.subheader("Backtest Metrics (2023-2025 Classes)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R²", "0.276")
    with col2:
        st.metric("RMSE", "0.154")
    with col3:
        st.metric("Players Tested", "34")
    
    st.subheader("Biggest Hits & Misses")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Best Predictions**")
        best = backtest.nsmallest(5, 'abs_error')[['player', 'draft_year', 'RIS', 'predicted_RIS', 'abs_error']]
        best.columns = ['Player', 'Year', 'Actual RIS', 'Predicted RIS', 'Error']
        best = best.reset_index(drop=True)
        best.index += 1
        st.dataframe(best, use_container_width=True)
    
    with col2:
        st.markdown("**❌ Biggest Misses**")
        worst = backtest.nlargest(5, 'abs_error')[['player', 'draft_year', 'RIS', 'predicted_RIS', 'abs_error']]
        worst.columns = ['Player', 'Year', 'Actual RIS', 'Predicted RIS', 'Error']
        worst = worst.reset_index(drop=True)
        worst.index += 1
        st.dataframe(worst, use_container_width=True)
    
    st.caption("Model trained on 2019-2022 classes, tested on 2023-2025 classes")

st.markdown("---")

# Player explorer
st.markdown("---")

# Player Explorer
st.header("Player Explorer")

# Player status notes
player_status = {
    "Cotie McMahon": "⚠️ Out for season — UCL tear",
    "Taina Mair": "⚠️ Waived before playing a game",
}

selected_player = st.selectbox("Select a player", predictions['player'].sort_values())

# Show status warning if applicable
if selected_player in player_status:
    st.warning(player_status[selected_player])

# Load validation data
validation = pd.read_csv("data/processed/validation_snapshot.csv")
player_pred = predictions[predictions['player'] == selected_player].iloc[0]
player_val = validation[validation['player'] == selected_player]

# Top metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Team", player_pred['wnba_team'])
with col2:
    st.metric("Draft Pick", f"#{int(player_pred['draft_pick'])}")
with col3:
    st.metric("Predicted RIS", f"{player_pred['predicted_RIS']:.3f}")
with col4:
    if not player_val.empty and not pd.isna(player_val['actual_RIS'].values[0]):
        actual = player_val['actual_RIS'].values[0]
        delta = round(actual - player_pred['predicted_RIS'], 3)
        st.metric("Actual RIS", f"{actual:.3f}", delta=f"{delta:+.3f}")
    else:
        st.metric("Actual RIS", "No data yet")

# Current season stats
if not player_val.empty and not pd.isna(player_val['G'].values[0]):
    st.subheader("2026 Season Stats")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Games", int(player_val['G'].values[0]))
    with col2:
        st.metric("PPG", player_val['ppg'].values[0])
    with col3:
        st.metric("RPG", player_val['rpg'].values[0])
    with col4:
        st.metric("APG", player_val['apg'].values[0])
    with col5:
        st.metric("MPG", player_val['mpg'].values[0])
else:
    st.info("No current season stats available for this player.")

# Predicted vs Actual comparison chart
if not player_val.empty and not pd.isna(player_val['actual_RIS'].values[0]):
    st.subheader("Predicted vs Actual RIS")
    
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(['Actual RIS', 'Predicted RIS'],
                   [player_val['actual_RIS'].values[0], player_pred['predicted_RIS']],
                   color=['#C8102E', '#418FDE'], edgecolor='white', linewidth=0.5)
    
    for bar, val in zip(bars, [player_val['actual_RIS'].values[0], player_pred['predicted_RIS']]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='white', fontsize=12)
    
    ax.set_xlim(0, 1.0)
    ax.set_facecolor('#0E1117')
    fig.patch.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    
    st.pyplot(fig)
    st.caption("⚠️ Early season — actual RIS based on 2-3 games only. Will stabilize over time.")

    st.markdown("---")

# Rookie Leaderboard
st.header("2026 Rookie Leaderboard")
st.caption("Among rookies with available stats — updated May 15, 2026")

# Get players with stats
leaderboard = validation[validation['G'].notna() & 
                          validation['ppg'].notna()].copy()

# Remove waived/injured
leaderboard = leaderboard[~leaderboard['player'].isin(['Cotie McMahon', 'Taina Mair'])]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Points Per Game")
    ppg_df = leaderboard[['player', 'ppg']].sort_values('ppg', ascending=False).reset_index(drop=True)
    ppg_df.index += 1
    ppg_df.columns = ['Player', 'PPG']
    st.dataframe(ppg_df, use_container_width=True)

    st.subheader("Assists Per Game")
    apg_df = leaderboard[['player', 'apg']].sort_values('apg', ascending=False).reset_index(drop=True)
    apg_df.index += 1
    apg_df.columns = ['Player', 'APG']
    st.dataframe(apg_df, use_container_width=True)

with col2:
    st.subheader("Rebounds Per Game")
    rpg_df = leaderboard[['player', 'rpg']].sort_values('rpg', ascending=False).reset_index(drop=True)
    rpg_df.index += 1
    rpg_df.columns = ['Player', 'RPG']
    st.dataframe(rpg_df, use_container_width=True)

    st.subheader("Minutes Per Game")
    mpg_df = leaderboard[['player', 'mpg']].sort_values('mpg', ascending=False).reset_index(drop=True)
    mpg_df.index += 1
    mpg_df.columns = ['Player', 'MPG']
    st.dataframe(mpg_df, use_container_width=True)

st.markdown("---")
st.caption("Built by Charlize Andaya | Data: Sports Reference & Basketball Reference | Model: Ridge Regression")