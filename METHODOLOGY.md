# Methodology & Technical Documentation

## Project Overview
This project builds a predictive model to estimate the rookie-year WNBA impact of 2026 first-round draft picks, based on their NCAA career statistics and team context. The central question is: **did teams pick the right players, and who is likely to have the most impact?**

---

## Step 1: Data Collection

### NCAA Prospect Stats
Per-game career statistics were manually collected from Sports Reference (sports-reference.com/cbb) for:
- All 12 NCAA players selected in Round 1 of the 2026 WNBA Draft
- 86 historical NCAA players selected in Round 1 of the 2019–2025 WNBA Drafts who actually played at least one WNBA game

International players (e.g. Awa Fam, Nell Angloma) were excluded because their stats are not comparable to NCAA data.

### WNBA Rookie Stats
Per-game statistics for the first WNBA season of each historical draftee were collected from Basketball Reference (basketball-reference.com/wnba), covering 2019–2025 draft classes, Round 1 only.

### WNBA Team Stats
Per-game, advanced, and shooting statistics for each WNBA team from 2019–2025 were collected from Basketball Reference. These capture team context — pace, offensive/defensive efficiency, scoring — that a rookie enters into.

---

## Step 2: Data Cleaning

All datasets went through the following cleaning steps:

- **Blank rows dropped** — Sports Reference CSVs include empty separator rows
- **Career summary rows separated** — the "Career" totals row was kept separately and not used as a season observation
- **Transfer notation rows removed** — rows like "Stanford (1 Yr)" indicating transfer history were dropped
- **Missing percentages filled with 0** — players with 0 attempts in a category (e.g. Lauren Betts with no 3-point attempts) had NaN percentages filled with 0
- **Seasons where G = NaN dropped** — injury/redshirt seasons with no games played were removed (e.g. Olivia Miles 2023-24, Azzi Fudd 2023-24)
- **Numeric columns converted** — all stat columns converted from string to float
- **Team names standardized** — playoff asterisks (*) removed from team names for clean merging

---

## Step 3: Feature Engineering

### Rookie Impact Score (RIS)
The target variable is a composite metric called the Rookie Impact Score, built from each historical rookie's first WNBA season. It is designed to capture both availability and production:

| Component | Weight | Reasoning |
|-----------|--------|-----------|
| Games played | 20% | Availability and roster security — making the team matters |
| Minutes per game | 20% | Coaching trust — getting on the court consistently |
| Points per game | 20% | Scoring production |
| Rebounds per game | 15% | Physical impact |
| Assists per game | 10% | Playmaking contribution |
| Win Shares per 40 min | 15% | Overall efficiency |

**Why these weights?**
We chose roughly equal weights across the first three components because we had no strong prior reason to favor one over another. Minutes and games played were included alongside raw stats because a player averaging 20 PPG in 5 minutes of garbage time is very different from one contributing 12 PPG as a starter. WS/40 captures efficiency and was weighted at 15% to reward players who produce without wasting possessions.

**Limitation:** These weights were chosen by judgment, not learned from data. A more rigorous approach would use regression or optimization to learn the weights that best predict long-term WNBA success. This is a noted area for future improvement.

All six components were normalized to a 0–1 scale using MinMaxScaler before weighting, so no single component dominates due to scale differences.

### Weighted Career Averages
Rather than using just the final season or a simple average, each prospect's career stats were averaged with **recency weighting** — more recent seasons receive higher weight. For a player with 4 seasons, the weights are 1/10, 2/10, 3/10, 4/10.

**Why recency weighting?**
A player's most recent season is the most representative of where they are as a player entering the draft. A freshman year from 4 years ago is less predictive than their senior season.

### Team Context Features
Each historical rookie was matched to their drafting team's stats from the season they were drafted into. For 2026 prospects, the 2025 team stats were used as a proxy for current team context.

**Toronto Tempo exception:** As a 2026 expansion team with no prior stats, Toronto Tempo was assigned 2025 league average values across all team features. This is a known limitation — expansion teams are inherently unpredictable.

---

## Step 4: Modeling

### Model Selection
Three models were evaluated using 5-fold cross validation:

| Model | R² | RMSE |
|-------|-----|------|
| Ridge Regression | 0.172 | 0.145 |
| Random Forest | 0.077 | 0.157 |
| Gradient Boosting | -0.136 | 0.171 |

Ridge Regression performed best and was selected as the final model.

**Why did Ridge outperform Random Forest and Gradient Boosting?**
With only 86 training samples, complex models like Random Forest and Gradient Boosting are prone to overfitting — they learn the training data too well and fail to generalize to new players. Ridge Regression is a simpler linear model with a regularization penalty (alpha) that prevents overfitting by shrinking coefficients toward zero. In small dataset scenarios, simpler models almost always outperform complex ones because there isn't enough data for complex models to learn meaningful patterns without also learning noise. This is a well-documented phenomenon in machine learning called the bias-variance tradeoff — with small samples, you want higher bias and lower variance, which is exactly what Ridge provides.

### What is R²?
R² (R-squared) measures how much of the variance in rookie success the model explains. An R² of 0.172 means the model explains approximately 17% of the variance in RIS. The remaining 83% is driven by factors not in our data — coaching decisions, role fit, injuries, player development, and team chemistry.

**Is 0.172 good?**
In the context of sports prediction, yes — this is within the normal range for models predicting individual athletic performance. Predicting human performance is inherently noisy. Published sports analytics models typically achieve R² between 0.10 and 0.35 for similar problems. The value of this model is not that it perfectly predicts outcomes, but that it systematically surfaces patterns that pure scouting intuition might miss.

**Leave-One-Out Cross Validation** was also performed as a more robust alternative for small datasets. LOO R² was 0.418 with RMSE of 0.144 — higher than 5-fold CV because each fold trains on 85 of 86 players rather than ~69. The true model performance likely falls between these estimates (R² 0.25–0.35), and the gap between them confirms that model performance would improve meaningfully with more training data.

### Feature Importance
The three most predictive features were:
1. **College points per game** — the strongest single predictor of WNBA rookie success
2. **Draft pick position** — nearly equally predictive, confirming that scouts add real value
3. **Games played in college** — durability and availability in college translates to the pro level

Notably, **3-point percentage and turnovers had almost no predictive value** — suggesting that shooting from distance and ball security in college don't translate meaningfully to WNBA rookie impact.

### Why did we add draft pick as a feature?
Without draft pick, the model R² was only 0.012 — essentially random. Adding draft pick position improved R² to 0.172. This makes intuitive sense: WNBA scouts and GMs have access to far more information than per-game statistics (film, athleticism, character, medical history, etc.). Draft position encodes that collective judgment. Our model adds value on top of that signal using statistical patterns that scouts may not systematically track.

---

## Step 5: Validation

### Historical Backtest
The model was backtested on 2023–2025 draft classes (34 players) after being trained on 2019–2022 classes. Backtest metrics:

| Metric | Value |
|--------|-------|
| R² | 0.276 |
| RMSE | 0.154 |
| Players Tested | 34 |

**Injury-affected seasons** are flagged in the dashboard and excluded from the "Biggest Misses" ranking, as a low actual RIS caused by injury is not a model failure. Affected players:
- **Nika Mühl (2024)** — missed the season with an ACL tear
- **Cotie McMahon (2026)** — suffered a season-ending UCL tear before playing a game

### 2026 Live Validation
Predictions are validated against real 2026 WNBA season data as it unfolds. The validation notebook pulls current per-game statistics and compares them against predicted RIS scores. Updates are planned every 2 weeks throughout the 2026 season.

Early validation snapshots are available in `data/processed/validation_snapshot.csv`.

---

## Step 6: Dashboard

The live dashboard was built with Streamlit and deployed at [wnba-draft-fit-2026.streamlit.app](https://wnba-draft-fit-2026.streamlit.app/).

### Interactive Charts
Static matplotlib visualizations were converted to interactive Plotly charts for two key tabs:
- **Predicted RIS bar chart** — color-coded by WNBA team, hover shows team, pick number, and predicted score
- **Draft Position vs Impact scatter** — hover shows player name, team, pick, and predicted RIS
- **Historical Accuracy scatter** — hover shows player name, actual vs predicted RIS, pick number, and error; color-coded by class year

### Historical Accuracy Section
The Historical Accuracy tab is structured as:
1. Interactive scatter plot (predicted vs actual RIS, color-coded by class year)
2. Backtest metrics (R², RMSE, players tested)
3. Biggest Hits & Misses spotlight — injury-affected players excluded from misses with an expandable note explaining why
4. Results by class year — tabbed breakdown for 2023, 2024, and 2025 with injury flags inline

### 2026 Season Tracker
The bottom section of the dashboard groups the Player Explorer and Rookie Leaderboard under a shared "2026 Season Tracker" header, making clear that both tools are live validation features updated throughout the season.

---

## Known Limitations

1. **Small training set** — 86 players is a small sample for machine learning. More historical data would improve model reliability.
2. **International players excluded** — players without NCAA careers cannot be evaluated by this model.
3. **Round 1 only** — the model focuses on first-round picks where stakes are highest and data quality is best.
4. **Injury history not modeled** — players like Azzi Fudd who missed significant college time due to injury may be systematically undervalued, as their weighted averages don't reflect their true ceiling.
5. **Team context is approximate** — using prior-year team stats as a proxy for current team needs ignores offseason roster moves.
6. **RIS weights are judgment-based** — future work could learn optimal weights from data rather than assigning them manually.
7. **Position not included** — centers and guards translate differently from college to the WNBA; adding position as a feature could improve fit predictions.
8. **Conference strength not adjusted** — a 20 PPG scorer in the ACC is treated the same as 20 PPG in a weaker conference.

---

## Roadmap

### Completed
- Ridge Regression model trained on 86 historical rookies (2019–2022)
- Leave-one-out cross validation for more robust performance estimation
- Historical backtest on 2023–2025 classes (R² = 0.276)
- Interactive Streamlit dashboard deployed at [wnba-draft-fit-2026.streamlit.app](https://wnba-draft-fit-2026.streamlit.app/)
- Interactive Plotly charts replacing static images for key visualizations
- Injury-aware historical accuracy section with class year tabs
- 2026 Season Tracker with live player explorer and rookie leaderboard

### Currently In Progress
- Midseason validation updates every 2 weeks throughout the 2026 season

### Planned
- Historical comp finder — identifying the most statistically similar historical rookie for each 2026 prospect

### Future Improvements

#### Modeling
- **Learn RIS weights from data** — use optimization or regression to find weights that best predict long-term WNBA success
- **Add player position as a feature** — position-specific models could improve accuracy
- **Conference strength adjustment** — normalize college stats by strength of schedule
- **Expand training data to rounds 2 and 3** — would roughly triple the training set
- **Build a Bayesian model** — use draft position as a prior, update with statistical evidence
- **Try ensemble methods** — combine predictions from multiple models

#### Data
- **Include international players** — build a separate model using EuroLeague or FIBA stats with a translation layer
- **Add injury history** — flag players who missed significant college time
- **Add physical measurables** — height, wingspan, athleticism from pre-draft workouts
- **Track team roster turnover** — opportunity matters as much as fit
- **Add coaching history** — certain coaches are known for developing young players

#### Validation & Monitoring
- **Automated validation pipeline** — pull updated stats automatically
- **Confidence intervals** — produce ranges of likely outcomes instead of point predictions
- **Track prediction drift** — monitor how predictions change as more season data accumulates

#### Visualization
- **Radar/spider charts** — show each prospect's full statistical profile
- **Side-by-side player comparisons** — given any two prospects, show how their profiles compare
- **Historical comp finder** — for each 2026 prospect, identify the most statistically similar historical rookie