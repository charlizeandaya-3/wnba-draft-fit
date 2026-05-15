import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="WNBA Draft Fit Predictor",
    page_icon="🏀",
    layout="wide"
)

# Global font size
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

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
    team_colors = {
        'Washington Mystics': '#002B5C',
        'Atlanta Dream': '#C8102E',
        'Toronto Tempo': '#75233B',
        'Seattle Storm': '#2C5234',
        'Chicago Sky': '#418FDE',
        'Minnesota Lynx': '#236192',
        'Dallas Wings': '#C4D600',
        'Indiana Fever': '#E03A3E',
        'Connecticut Sun': '#F77F00',
    }

    plot_df = predictions[['player', 'wnba_team', 'draft_pick', 'predicted_RIS']].sort_values('predicted_RIS', ascending=True)
    plot_df['color'] = plot_df['wnba_team'].map(lambda t: team_colors.get(t, '#888888'))

    fig1 = go.Figure()

    fig1.add_trace(go.Bar(
        x=plot_df['predicted_RIS'],
        y=plot_df['player'],
        orientation='h',
        marker=dict(
            color=plot_df['color'],
            line=dict(color='white', width=0.5)
        ),
        customdata=plot_df[['wnba_team', 'draft_pick']].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Team: %{customdata[0]}<br>"
            "Pick: #%{customdata[1]}<br>"
            "Predicted RIS: %{x:.3f}<extra></extra>"
        )
    ))

    fig1.update_layout(
        title=dict(
            text='2026 WNBA Draft Class — Predicted Rookie Impact Score<br>'
                 '<sup>By Team Fit & NCAA Career Profile</sup>',
            font=dict(size=24),
        ),
        xaxis=dict(
            title='Predicted Rookie Impact Score (RIS)',
            title_font=dict(size=16),
            tickfont=dict(size=14),
            range=[0, 0.75],
        ),
        yaxis=dict(
            tickfont=dict(size=14),
        ),
        template='plotly_dark',
        height=550,
        showlegend=False,
    )

    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    predictions_plot = pd.read_csv("data/processed/predictions_2026.csv")

    x_ref = np.linspace(1, 15, 100)
    y_ref = 0.65 - (x_ref - 1) * (0.65 - 0.25) / 14

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=x_ref, y=y_ref,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Expected (if draft order = impact order)',
        hoverinfo='skip'
    ))

    fig2.add_trace(go.Scatter(
        x=predictions_plot['draft_pick'],
        y=predictions_plot['predicted_RIS'],
        mode='markers',
        marker=dict(size=12, color='#418FDE', line=dict(color='white', width=0.5)),
        text=predictions_plot['player'],
        customdata=predictions_plot[['wnba_team', 'draft_pick']].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Team: %{customdata[0]}<br>"
            "Pick: #%{customdata[1]}<br>"
            "Predicted RIS: %{y:.3f}<extra></extra>"
        ),
        name='2026 Picks'
    ))

    fig2.update_layout(
        title=dict(
            text='2026 WNBA Draft — Pick Position vs Predicted Impact<br>'
                 '<sup>Players above the line are undervalued; below are overvalued</sup>',
            font=dict(size=24),
        ),
        xaxis=dict(
            title='Draft Pick Number',
            title_font=dict(size=16),
            tickfont=dict(size=14),
            dtick=1,
        ),
        yaxis=dict(
            title='Predicted Rookie Impact Score (RIS)',
            title_font=dict(size=16),
            tickfont=dict(size=14),
        ),
        legend=dict(font=dict(size=14)),
        template='plotly_dark',
        height=550,
    )

    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.image("visuals/feature_importance.png", use_container_width=True)

with tab4:
    st.image("visuals/ris_distribution.png", use_container_width=True)

with tab5:
    backtest = pd.read_csv("data/processed/backtest_results.csv")

    injury_exclusions = {
        "Nika Mühl": "Missed 2024 season with ACL tear",
        "Cotie McMahon": "Missed 2026 season with UCL tear",
    }

    colors = {2023: '#418FDE', 2024: '#C8102E', 2025: '#2C5234'}
    fig = go.Figure()

    for year in [2023, 2024, 2025]:
        year_df = backtest[backtest['draft_year'] == year]
        fig.add_trace(go.Scatter(
            x=year_df['RIS'],
            y=year_df['predicted_RIS'],
            mode='markers',
            name=str(year),
            marker=dict(color=colors[year], size=10, line=dict(color='white', width=0.5)),
            text=year_df['player'],
            customdata=year_df[['draft_pick', 'abs_error']].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Actual RIS: %{x:.3f}<br>"
                "Predicted RIS: %{y:.3f}<br>"
                "Pick #%{customdata[0]}<br>"
                "Error: %{customdata[1]:.3f}<extra></extra>"
            )
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Perfect prediction',
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=dict(
            text='Historical Accuracy — Predicted vs Actual RIS<br>'
                 '<sup>2023-2025 Draft Classes</sup>',
            font=dict(size=24),
        ),
        xaxis=dict(
            title='Actual RIS',
            title_font=dict(size=16),
            tickfont=dict(size=14),
        ),
        yaxis=dict(
            title='Predicted RIS',
            title_font=dict(size=16),
            tickfont=dict(size=14),
        ),
        legend=dict(x=0.01, y=0.99, font=dict(size=14)),
        template='plotly_dark',
        height=550,
    )

    st.plotly_chart(fig, use_container_width=True)

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
        best = backtest.nsmallest(5, 'abs_error')[
            ['player', 'draft_year', 'RIS', 'predicted_RIS', 'abs_error']
        ]
        best.columns = ['Player', 'Year', 'Actual RIS', 'Predicted RIS', 'Error']
        best = best.reset_index(drop=True)
        best.index += 1
        st.dataframe(best, use_container_width=True)

    with col2:
        st.markdown("**❌ Biggest Misses** _(injury-affected seasons excluded)_")
        backtest_no_injury = backtest[~backtest['player'].isin(injury_exclusions.keys())]
        worst = backtest_no_injury.nlargest(5, 'abs_error')[
            ['player', 'draft_year', 'RIS', 'predicted_RIS', 'abs_error']
        ]
        worst.columns = ['Player', 'Year', 'Actual RIS', 'Predicted RIS', 'Error']
        worst = worst.reset_index(drop=True)
        worst.index += 1
        st.dataframe(worst, use_container_width=True)

        with st.expander("ℹ️ Excluded from misses"):
            for player, reason in injury_exclusions.items():
                st.markdown(f"- **{player}**: {reason}")

    st.caption("Model trained on 2019-2022 classes, tested on 2023-2025 classes")

    st.subheader("Results by Class Year")
    class_tabs = st.tabs(["2023", "2024", "2025"])

    for tab, year in zip(class_tabs, [2023, 2024, 2025]):
        with tab:
            class_df = backtest[backtest['draft_year'] == year][
                ['player', 'RIS', 'predicted_RIS', 'abs_error']
            ].copy()
            class_df.columns = ['Player', 'Actual RIS', 'Predicted RIS', 'Error']
            class_df = class_df.sort_values('Error').reset_index(drop=True)
            class_df.index += 1

            class_df['Note'] = class_df['Player'].map(
                lambda p: "⚠️ Injury" if p in injury_exclusions else ""
            )

            st.dataframe(class_df, use_container_width=True)

            injured_in_class = [
                f"**{p}**: {r}" for p, r in injury_exclusions.items()
                if p in class_df['Player'].values
            ]
            if injured_in_class:
                for note in injured_in_class:
                    st.caption(f"⚠️ {note}")

st.markdown("---")

# 2026 Season Tracker
st.header("2026 Season Tracker")
st.caption("Live validation — tracking how 2026 predictions hold up as the season unfolds.")

# Player Explorer
st.subheader("Player Explorer")

player_status = {
    "Cotie McMahon": "⚠️ Out for season — UCL tear",
    "Taina Mair": "⚠️ Waived before playing a game",
}

selected_player = st.selectbox("Select a player", predictions['player'].sort_values())

if selected_player in player_status:
    st.warning(player_status[selected_player])

validation = pd.read_csv("data/processed/validation_snapshot.csv")
player_pred = predictions[predictions['player'] == selected_player].iloc[0]
player_val = validation[validation['player'] == selected_player]

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
st.subheader("Rookie Leaderboard")
st.caption("Among rookies with available stats — updated May 15, 2026")

leaderboard = validation[validation['G'].notna() &
                          validation['ppg'].notna()].copy()
leaderboard = leaderboard[~leaderboard['player'].isin(['Cotie McMahon', 'Taina Mair'])]

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Points Per Game**")
    ppg_df = leaderboard[['player', 'ppg']].sort_values('ppg', ascending=False).reset_index(drop=True)
    ppg_df.index += 1
    ppg_df.columns = ['Player', 'PPG']
    st.dataframe(ppg_df, use_container_width=True)

    st.markdown("**Assists Per Game**")
    apg_df = leaderboard[['player', 'apg']].sort_values('apg', ascending=False).reset_index(drop=True)
    apg_df.index += 1
    apg_df.columns = ['Player', 'APG']
    st.dataframe(apg_df, use_container_width=True)

with col2:
    st.markdown("**Rebounds Per Game**")
    rpg_df = leaderboard[['player', 'rpg']].sort_values('rpg', ascending=False).reset_index(drop=True)
    rpg_df.index += 1
    rpg_df.columns = ['Player', 'RPG']
    st.dataframe(rpg_df, use_container_width=True)

    st.markdown("**Minutes Per Game**")
    mpg_df = leaderboard[['player', 'mpg']].sort_values('mpg', ascending=False).reset_index(drop=True)
    mpg_df.index += 1
    mpg_df.columns = ['Player', 'MPG']
    st.dataframe(mpg_df, use_container_width=True)

st.markdown("---")
st.caption("Built by Charlize Andaya | Data: Sports Reference & Basketball Reference | Model: Ridge Regression")