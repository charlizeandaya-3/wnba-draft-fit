# WNBA Draft Fit Predictor
🔗 **[Live Dashboard](https://wnba-draft-fit-2026.streamlit.app/)**

Predicting which 2026 WNBA Round 1 draft picks will have the most rookie impact — and whether teams made the right call.

## The Question
Did WNBA teams pick the right players in the 2026 draft? This project builds a data-driven model to predict rookie impact based on NCAA career performance and team context, then validates predictions against real 2026 season data as it unfolds.

## Key Finding
Lauren Betts (pick #4, Washington Mystics) projects as the highest-impact rookie of the 2026 class. Azzi Fudd, despite going #1 overall to Dallas, projects below her draft position based on her college statistical profile relative to team fit. Madina Okot is the most undervalued pick in the class — selected 13th by Atlanta, she projects as the second-highest impact rookie.

## Methodology

### Target Variable — Rookie Impact Score (RIS)
A composite metric built from 2019–2025 Round 1 rookie seasons:
- Games played (20%) — availability and roster security
- Minutes per game (20%) — coaching trust
- Points per game (20%) — scoring production
- Rebounds per game (15%) — physical impact
- Assists per game (10%) — playmaking
- Win Shares per 40 minutes (15%) — overall efficiency

### Model
Ridge Regression trained on 86 historical Round 1 rookies (2019–2025) using weighted NCAA career averages and draft position. Validated with 5-fold cross validation (R² = 0.172) and leave-one-out cross validation (R² = 0.418). True model performance likely falls between these estimates.

### Features
- Weighted NCAA career averages (scoring, rebounding, assists, efficiency, shooting)
- Draft pick position
- Team offensive/defensive context (offensive rating, pace, scoring)

## Data Sources
- [Sports Reference CBB](https://www.sports-reference.com/cbb/) — NCAA career stats
- [Basketball Reference WNBA](https://www.basketball-reference.com/wnba/) — WNBA rookie and team stats

## 2026 Predictions

| Player | Team | Pick | Predicted RIS |
|--------|------|------|--------------|
| Lauren Betts | Washington Mystics | 4 | 0.595 |
| Madina Okot | Atlanta Dream | 13 | 0.521 |
| Kiki Rice | Toronto Tempo | 6 | 0.496 |
| Flau'jae Johnson | Seattle Storm | 8 | 0.486 |
| Gabriela Jaquez | Chicago Sky | 5 | 0.481 |
| Olivia Miles | Minnesota Lynx | 2 | 0.476 |
| Azzi Fudd | Dallas Wings | 1 | 0.414 |
| Raven Johnson | Indiana Fever | 10 | 0.333 |
| Taina Mair | Seattle Storm | 14 | 0.326 |
| Gianna Kneepkens | Connecticut Sun | 15 | 0.301 |
| Angela Dugalic | Washington Mystics | 9 | 0.300 |
| Cotie McMahon | Washington Mystics | 11 | 0.288 |

## Dashboard
The live Streamlit dashboard includes:
- **Interactive predicted RIS bar chart** — color-coded by WNBA team, with hover showing team, pick, and predicted score
- **Interactive draft position vs impact chart** — shows which players are above/below expected value for their pick slot
- **Historical accuracy section** — backtest on 2023–2025 classes (R² = 0.276), with injury-affected players flagged and results broken out by class year
- **2026 Season Tracker** — player explorer with live stats and predicted vs actual RIS comparison, plus a rookie leaderboard updated throughout the season

## Key Insights

**Scoring and draft position are the strongest predictors of rookie success.**
College points per game and draft pick number are nearly equally predictive of WNBA rookie impact, suggesting scouts are doing their job but raw scoring still adds signal on top of draft position alone.

**Games played in college matters more than shooting efficiency.**
Availability and durability ranks third in feature importance — players who stayed healthy and played consistently in college tend to translate better to the pros. Three-point percentage and turnovers are nearly irrelevant predictors.

**Madina Okot is the most undervalued pick in the 2026 draft.**
Selected 13th overall by Atlanta, her dominant rebounding profile (10.3 RPG in college) projects her as the second-highest impact rookie in the class.

**Azzi Fudd may be the most overvalued pick.**
Despite going #1 overall to Dallas, her predicted RIS ranks 7th in the class. Her injury-shortened junior year likely suppresses her weighted career averages — this may underestimate her true ceiling, but based purely on college production relative to draft position, she projects below expectations.

**Washington Mystics had a split draft.**
With three first-round picks, Washington's haul was uneven — Lauren Betts projects as the top rookie in the entire class, but Angela Dugalic and Cotie McMahon both project near the bottom. McMahon had a partial UCL tear before playing a game, but returned for her first game on May 15th.

## Model Limitations
- Small training set (86 players) limits model power
- Only NCAA players included — international prospects excluded due to lack of comparable college stats
- Only Round 1 picks analyzed
- Injury history not modeled — players with significant missed time are systematically undervalued
- Toronto Tempo team context approximated using 2025 league average (expansion team)
- Taina Mair was waived before playing, but was signed as a development player soon after; Cotie McMahon had a partial UCL tear — both illustrate that roster decisions and injuries cannot be predicted from college stats alone.

## Project Structure

```
wnba-draft-fit/
├── data/
│   ├── external/        # Raw CSVs from Sports/Basketball Reference
│   └── processed/       # Cleaned and merged datasets
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_cleaning_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_validation.ipynb
├── visuals/             # Generated chart images
├── src/
│   ├── scraper.py
│   ├── features.py
│   └── model.py
├── app.py               # Streamlit dashboard
└── README.md
```

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn, plotly, Streamlit, Jupyter, GitHub

## Status
🟡 In Progress — 2026 season underway, validation ongoing, dashboard live

## Automated Updates
Validation data updates automatically every Monday via GitHub Actions. The scraper pulls current WNBA per-game and advanced stats from Basketball Reference, recomputes actual RIS for all 2026 rookies, and commits the updated CSVs to the repo. The Streamlit dashboard reflects the latest data on next page load.