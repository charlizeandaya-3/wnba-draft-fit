import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import date
import warnings
warnings.filterwarnings('ignore')

def scrape_bball_ref(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    table = soup.find('table')
    
    # Get all th data-stats including duplicates
    col_names = [th.get('data-stat') for th in table.find_all('th') if th.get('data-stat')]
    # Don't deduplicate — keep all
    
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        if 'thead' in tr.get('class', []):
            continue
        th = tr.find('th')
        tds = tr.find_all('td')
        if not th or not tds:
            continue
        a = th.find('a')
        player_name = a.get_text(strip=True) if a else th.get_text(strip=True)
        cells = [player_name] + [td.get_text(strip=True) for td in tds]
        rows.append(cells)
    
    if not rows:
        return pd.DataFrame()
    
    # Use row length to slice col_names
    row_len = len(rows[0])
    cols = col_names[:row_len]
    
    # Make column names unique if needed
    seen = {}
    unique_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            unique_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            unique_cols.append(c)
    
    df = pd.DataFrame(rows, columns=unique_cols)
    df = df[df['player'] != 'Player'].reset_index(drop=True)
    return df
# ----------------------------
# Step 1: Scrape per-game stats
# ----------------------------
print("Scraping per-game stats...")
per_game = scrape_bball_ref(
    "https://www.basketball-reference.com/wnba/years/2026_per_game.html"
)
per_game.to_csv("data/external/wnba_2026_current_stats.csv", index=False)
print(f"Saved wnba_2026_current_stats.csv — {len(per_game)} rows")
print(per_game[['player', 'g', 'pts_per_g']].head())

# ----------------------------
# Step 2: Scrape advanced stats
# ----------------------------
print("Scraping advanced stats...")
advanced = scrape_bball_ref(
    "https://www.basketball-reference.com/wnba/years/2026_advanced.html"
)
advanced.to_csv("data/external/wnba_2026_advanced_stats.csv", index=False)
print(f"Saved wnba_2026_advanced_stats.csv — {len(advanced)} rows")
print("Advanced cols:", advanced.columns.tolist())

# ----------------------------
# Step 3: Normalize player names
# ----------------------------
def normalize_name(series):
    return (
        series.str.normalize('NFKD')
        .str.encode('ascii', errors='ignore')
        .str.decode('ascii')
        .str.strip()
    )

per_game['player'] = normalize_name(per_game['player'])
advanced['player'] = normalize_name(advanced['player'])

# Fix Dugalic specifically
per_game['player'] = per_game['player'].str.replace('DugaliA', 'Dugalic', regex=False)
advanced['player'] = advanced['player'].str.replace('DugaliA', 'Dugalic', regex=False)

# ----------------------------
# Step 4: Convert numeric columns
# ----------------------------
# Step 4
for col in ['g', 'mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g']:
    per_game[col] = pd.to_numeric(per_game[col], errors='coerce')
    advanced['ws_per_40'] = pd.to_numeric(advanced['ws_per_40'], errors='coerce')

# ----------------------------
# Step 5: Load predictions
# ----------------------------
predictions = pd.read_csv("data/processed/predictions_2026.csv")
rookie_names = predictions['player'].tolist()

# ----------------------------
# Step 6: Filter to 2026 rookies
# ----------------------------
current_rookies = per_game[per_game['player'].isin(rookie_names)].copy()
current_rookies = current_rookies.rename(columns={'Player': 'player'})
print(f"\nRookies found in per-game stats: {len(current_rookies)}")
print("Matched:", current_rookies['player'].tolist())
print("Missing:", [p for p in rookie_names if p not in current_rookies['player'].tolist()])
# Fix the print statement
print(current_rookies[['player', 'g', 'mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g']])

adv_rookies = advanced[advanced['player'].isin(rookie_names)][['player', 'ws_per_40']].copy()

# ----------------------------
# Step 7: Merge everything
# ----------------------------
# Fix the merge
validation_df = predictions[['player', 'wnba_team', 'draft_pick', 'predicted_RIS']].merge(
    current_rookies[['player', 'g', 'mp_per_g', 'pts_per_g', 'trb_per_g', 'ast_per_g']],
    on='player',
    how='left'
).merge(
    adv_rookies,
    on='player',
    how='left'
)

validation_df = validation_df.rename(columns={
    'g': 'G',
    'mp_per_g': 'mpg',
    'pts_per_g': 'ppg',
    'trb_per_g': 'rpg',
    'ast_per_g': 'apg',
    'ws_per_40': 'WS/40'
})
# ----------------------------
# Step 8: Compute actual RIS
# ----------------------------
def normalize(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)

has_stats = validation_df['G'].notna()

if has_stats.sum() > 0:
    ris_df = validation_df[has_stats].copy()

    max_games = 40
    ris_df['g_norm'] = (ris_df['G'] / max_games).clip(0, 1)
    ris_df['mpg_norm'] = normalize(ris_df['mpg'])
    ris_df['ppg_norm'] = normalize(ris_df['ppg'])
    ris_df['rpg_norm'] = normalize(ris_df['rpg'])
    ris_df['apg_norm'] = normalize(ris_df['apg'])
    ris_df['ws40_norm'] = normalize(ris_df['WS/40'])

    ris_df['actual_RIS'] = (
        ris_df['g_norm'] * 0.20 +
        ris_df['mpg_norm'] * 0.20 +
        ris_df['ppg_norm'] * 0.20 +
        ris_df['rpg_norm'] * 0.15 +
        ris_df['apg_norm'] * 0.10 +
        ris_df['ws40_norm'] * 0.15
    )

    validation_df = validation_df.merge(
        ris_df[['player', 'actual_RIS']],
        on='player',
        how='left'
    )
else:
    validation_df['actual_RIS'] = np.nan

# ----------------------------
# Step 9: Save
# ----------------------------
validation_df['snapshot_date'] = str(date.today())
validation_df['games_into_season'] = validation_df['G'].fillna(0).astype(int)

validation_df.to_csv("data/processed/validation_snapshot.csv", index=False)
print(f"\nSaved validation_snapshot.csv — {len(validation_df)} players")
print(validation_df[['player', 'predicted_RIS', 'G', 'ppg', 'rpg', 'apg', 'WS/40', 'actual_RIS']])