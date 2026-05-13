# WNBA Draft Fit Predictor

Predicting which 2026 WNBA Round 1 draft picks will have the most rookie impact — and whether teams made the right call.

## The Question
Did WNBA teams pick the right players in the 2026 draft? This project builds a data-driven model to predict rookie impact based on NCAA career performance and team context, then validates predictions against real 2026 season data as it unfolds.

## Key Finding
Lauren Betts (pick #4, Washington Mystics) projects as the highest-impact rookie of the 2026 class. Azzi Fudd, despite going #1 overall to Dallas, projects below her draft position based on her college statistical profile relative to team fit.

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
Ridge Regression trained on 86 historical Round 1 rookies (2019–2025) using weighted NCAA career averages and draft position. Validated with 5-fold cross validation (R² = 0.172).

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
├── src/
│   ├── scraper.py
│   ├── features.py
│   └── model.py
└── README.md
```

## Tools
Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter, GitHub

## Limitations
- Small training set (86 players) limits model power
- Only NCAA players included — international prospects (Awa Fam, Nell Angloma, etc.) excluded due to lack of comparable college stats
- Only Round 1 picks analyzed — model focuses on high-stakes picks where team fit matters most
- Players with significant injury history may be undervalued — for example, Azzi Fudd missed most of her junior year with a knee injury, so her weighted career averages don't fully reflect her ceiling
- Toronto Tempo team context approximated using 2025 league average (first-year expansion team with no prior stats)
- Early season validation data is limited — predictions will be updated throughout the 2026 season

## Status
In Progress — 2026 season underway, validation ongoing