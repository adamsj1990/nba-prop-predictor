import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import kagglehub
import requests
import sqlite3
import datetime
import scipy.stats as stats
from datetime import date, timedelta

# --- SECURE API & CONFIGURATION ---
# Keep your dataset safe in secrets; hardcode your live validated Odds API key directly below
try:
    os.environ["KAGGLE_API_TOKEN"] = st.secrets["KAGGLE_API_TOKEN"]
except KeyError:
    st.error("🚨 Configuration Error: KAGGLE_API_TOKEN missing from Streamlit Secrets!")
    st.stop()

THE_ODDS_API_KEY = "d46ad44217337fd9abcf115e59eaf01d"
DB_FILE = "props_history.db"

TEAM_MAPPING = {
    "BRK": "BKN",
    "PHO": "PHX",
    "CHO": "CHA",
}

# Map The Odds API unique team naming keys back to your local Kaggle dataset abbreviations
ODDS_API_TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

DEFENSIVE_YIELD_MULTIPLIER = {
    "PTS": {"CLE": 0.96, "DET": 1.04, "BKN": 1.02, "BOS": 0.92, "LAL": 1.01, "GSW": 0.99},
    "TRB": {"CLE": 0.94, "DET": 1.05, "BKN": 1.01, "BOS": 0.95, "LAL": 0.97, "GSW": 1.03},
    "AST": {"CLE": 0.98, "DET": 1.02, "BKN": 1.03, "BOS": 0.91, "LAL": 1.00, "GSW": 0.96},
    "3P":  {"CLE": 1.03, "DET": 1.01, "BKN": 0.99, "BOS": 0.88, "LAL": 1.02, "GSW": 1.01}
}

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_props (
            id TEXT PRIMARY KEY, saved_date TEXT, game_date TEXT, player TEXT,
            team TEXT, category TEXT, book_line REAL, xgboost_proj REAL,
            edge REAL, recommendation TEXT, actual_stat REAL, outcome TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

st.set_page_config(page_title="Sharp NBA Live Prop Predictor", layout="wide")
st.title("🧠 Sharp XGBoost Live Schedule Prop Board & Probability Distribution Engine")

def get_local_today():
    return (datetime.datetime.utcnow() - datetime.timedelta(hours=6)).date()

st.subheader("🗓️ Select Analysis Date")
if 'selected_date' not in st.session_state:
    st.session_state.selected_date = get_local_today()

cal_cols = st.columns(2)
for i in range(2):
    target_date = get_local_today() + timedelta(days=i)
    btn_label = f"🔥 TODAY\n({target_date.strftime('%m-%d')})" if i == 0 else f"⏳ TOMORROW\n({target_date.strftime('%m-%d')})"
    is_active = (st.session_state.selected_date == target_date)
    if cal_cols[i].button(btn_label, key=f"cal_{i}", use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state.selected_date = target_date
        st.rerun()

@st.cache_data(ttl=3600)
def load_data_from_kaggle():
    try:
        download_dir = kagglehub.dataset_download("eduardopalmieri/nba-player-stats-season-2526")
        target_csv_path = os.path.join(download_dir, 'nba_dailyleaders_full.csv')
        df = pd.read_csv(target_csv_path)
        df.columns = [c.upper() for c in df.columns]
        if 'TM' in df.columns: df['TM'] = df['TM'].str.strip().str.upper()
        if 'OPP' in df.columns: df['OPP'] = df['OPP'].str.strip().str.upper()
        if 'DATE' in df.columns: df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"🚨 Failed downloading/processing dataset from Kaggle Hub: {e}")
        return pd.DataFrame()

raw_df = load_data_from_kaggle()
if raw_df.empty: st.stop()

# --- THE ODDS API: SLATE INGESTION LOGIC ---
@st.cache_data(ttl=600)
def get_odds_api_games(api_key):
    """Hits the main sports directory endpoint to grab unique event IDs for matching."""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events?apiKey={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []

# --- THE ODDS API: GRANULAR PLAYER PROP INGESTION ---
@st.cache_data(ttl=300)
def fetch_odds_api_player_props(api_key, event_id):
    """Queries specific single-event metrics across all tracked US sportsbooks."""
    if not event_id: return []
    markets_csv = "player_points,player_rebounds,player_assists,player_threes"
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds?apiKey={api_key}&regions=us&markets={markets_csv}&oddsFormat=american"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get('bookmakers', [])
        return []
    except:
        return []

all_api_events = get_odds_api_games(THE_ODDS_API_KEY)
live_matchups_dict = {}

for event in all_api_events:
    commence_time = pd.to_datetime(event.get('commence_time')).date()
    if commence_time == st.session_state.selected_date:
        away = ODDS_API_TEAM_MAP.get(event.get('away_team'), "AWY")
        home = ODDS_API_TEAM_MAP.get(event.get('home_team'), "HOM")
        label = f"{away} @ {home}"
        live_matchups_dict[label] = event.get('id')

live_matchups = list(live_matchups_dict.keys())

if not live_matchups:
    st.info("ℹ️ No active game endpoints scheduled for this date via API. Loading backup matrix from dataset.")
    latest_date = raw_df['DATE'].max()
    slate_df = raw_df[raw_df['DATE'] == latest_date]
    matchup_set = set()
    for _, row in slate_df.iterrows():
        t1, t2 = row['TM'], row['OPP']
        if pd.notna(t1) and pd.notna(t2): matchup_set.add(tuple(sorted([t1, t2])))
    live_matchups = [f"{m[0]} @ {m[1]}" for m in matchup_set]

# --- SHARP MATH HELPER FUNCTIONS ---
def american_to_probability(odds):
    if odds is None or odds == 0: return 50.0
    return (abs(odds) / (abs(odds) + 100)) * 100 if odds < 0 else (100 / (odds + 100)) * 100

def calculate_projected_pace(away_team, home_team, raw_df):
    recent_data = raw_df[raw_df['DATE'] >= (raw_df['DATE'].max() - datetime.timedelta(days=30))]
    if recent_data.empty: return 100.0
    away_avg_fga = recent_data[recent_data['TM'] == away_team]['FGA'].median() * 5 / 48
    home_avg_fga = recent_data[recent_data['TM'] == home_team]['FGA'].median() * 5 / 48
    return (away_avg_fga + home_avg_fga) * 48 / 5

def generate_sharp_recommendation(proj, line, stat):
    edge = proj - line
    std_dev = 4.5 if stat == "PTS" else (2.5 if stat in ["TRB", "AST"] else 1.2)
    z_score = edge / std_dev
    model_over_prob = (1 - stats.norm.cdf(-z_score)) * 100
    if model_over_prob - 50.0 > 5.0: return "🟢 VALUE OVER", model_over_prob
    elif 50.0 - model_over_prob > 5.0: return "🔴 VALUE UNDER", (100 - model_over_prob)
    return "⚖️ HOLD (Efficient)", model_over_prob

def clean_minutes(val):
    if isinstance(val, str) and ':' in val:
        parts = val.split(':')
        return float(parts[0]) + (float(parts[1]) / 60)
    return pd.to_numeric(val, errors='coerce')

def generate_rationale(name, stat, proj, line, model_prob, total_mod, def_mod, source):
    direction = "OVER" if proj > line else "UNDER"
    env_impact = []
    if total_mod > 1.05: env_impact.append("elevated macro possession pacing")
    if def_mod > 1.02: env_impact.append("schematic defensive coverage leaks")
    return f"XGBoost engine isolates a mathematical variance using [{source}] data streams, assigning a {model_prob:.1f}% true probability toward the {direction}."

# --- INTERACTIVE INTERFACE ---
st.sidebar.header("🗓️ Select Matchup")
selected_matchup = st.sidebar.selectbox("Games on Selected Day", live_matchups)
away_team, home_team = selected_matchup.split(" @ ")

st.sidebar.header("🎰 Sportsbook Feed Source")
selected_book = st.sidebar.selectbox(
    "Active Oddsmaker Vendor",
    ["DraftKings", "FanDuel", "BetMGM", "Caesars", "LowVig"]
)
clean_book_str = selected_book.lower().strip()

target_event_id = live_matchups_dict.get(selected_matchup)
bookmakers_payload = fetch_odds_api_player_props(THE_ODDS_API_KEY, target_event_id)

# Map our internal keys to The Odds API market string filters
PROP_MAP_CATEGORIES = {
    "PTS": "player_points",
    "TRB": "player_rebounds",
    "AST": "player_assists",
    "3P": "player_threes"
}

st.sidebar.header("📊 Environmental Modifiers")
BASE_TOTAL = 218.0
if 'total_ou_val' not in st.session_state: st.session_state.total_ou_val = 218.0
if 'away_spread_val' not in st.session_state: st.session_state.away_spread_val = 0.0

st.sidebar.markdown("**Game Over/Under Line**")
ou_cols = st.sidebar.columns([1, 4, 1])
if ou_cols[0].button("➖", key="dec_ou", use_container_width=True):
    st.session_state.total_ou_val = max(195.0, st.session_state.total_ou_val - 0.5)
    st.rerun()
GAME_TOTAL_OU = ou_cols[1].slider("OU Slider", 195.0, 235.0, value=st.session_state.total_ou_val, step=0.5, label_visibility="collapsed")
st.session_state.total_ou_val = GAME_TOTAL_OU
if ou_cols[2].button("➕", key="inc_ou", use_container_width=True):
    st.session_state.total_ou_val = min(235.0, st.session_state.total_ou_val + 0.5)
    st.rerun()

st.sidebar.markdown(f"**{away_team} Spread**")
spread_cols = st.sidebar.columns([1, 4, 1])
if spread_cols[0].button("➖", key="dec_spread", use_container_width=True):
    st.session_state.away_spread_val = max(-15.0, st.session_state.away_spread_val - 0.5)
    st.rerun()
AWAY_SPREAD = spread_cols[1].slider("Spread Slider", -15.0, 15.0, value=st.session_state.away_spread_val, step=0.5, label_visibility="collapsed")
st.session_state.away_spread_val = AWAY_SPREAD
if spread_cols[2].button("➕", key="inc_spread", use_container_width=True):
    st.session_state.away_spread_val = min(15.0, st.session_state.away_spread_val + 0.5)
    st.rerun()

HOME_SPREAD = -AWAY_SPREAD

max_data_date = raw_df['DATE'].max()
roster_cutoff_date = max_data_date - datetime.timedelta(days=14)
active_rosters_df = raw_df[(raw_df['TM'].isin([away_team, home_team])) & (raw_df['DATE'] >= roster_cutoff_date)]
unique_players = active_rosters_df[['PLAYER', 'TM']].drop_duplicates()

results_list = []
total_modifier = GAME_TOTAL_OU / BASE_TOTAL
projected_game_pace = calculate_projected_pace(away_team, home_team, raw_df)
pace_modifier = projected_game_pace / 100.0
stats_to_analyze = ["PTS", "TRB", "AST", "3P"]

progress_bar = st.progress(0)
player_count = len(unique_players)

if player_count == 0:
    st.error(f"Could not find active rosters for {away_team} or {home_team} in the last 14 days.")
else:
    for idx, (_, p_row) in enumerate(unique_players.iterrows()):
        p_name = p_row['PLAYER']
        p_team = p_row['TM']
        p_spread = AWAY_SPREAD if p_team == away_team else HOME_SPREAD
        opp_team = home_team if p_team == away_team else away_team
        
        player_logs = raw_df[raw_df['PLAYER'] == p_name].copy()
        if len(player_logs) < 5: continue
        player_logs['MP'] = player_logs['MP'].apply(clean_minutes)
        player_logs = player_logs[player_logs['DATE'].dt.date < st.session_state.selected_date]
        if len(player_logs) < 5: continue
        
        for stat in stats_to_analyze:
            player_logs[stat] = pd.to_numeric(player_logs[stat], errors='coerce')
            player_logs['FGA'] = pd.to_numeric(player_logs['FGA'], errors='coerce')
            sub_df = player_logs.dropna(subset=[stat, 'MP', 'FGA']).copy()
            if len(sub_df) < 5: continue
            
            recent_logs = sub_df.sort_values('DATE', ascending=False).head(5)
            series_min_avg = recent_logs['MP'].mean()
            series_stat_avg = recent_logs[stat].mean()
            
            # --- THE ODDS API: PAYLOAD PARSING CORE ---
            real_line_found = False
            vegas_line, over_odds, under_odds = None, None, None
            line_source = f"🤖 Model Baseline"
            target_market_key = PROP_MAP_CATEGORIES[stat]
            
            # Loop 1: Strict User Bookmaker Selection
            for bookmaker in bookmakers_payload:
                if bookmaker.get('key', '') == clean_book_str:
                    for market in bookmaker.get('markets', []):
                        if market.get('key', '') == target_market_key:
                            for outcome in market.get('outcomes', []):
                                api_player = outcome.get('description', '')
                                # Handles name discrepancies (stripping punctuation/hyphens for security)
                                if p_name.lower().replace("-","").replace(" ","") in api_player.lower().replace("-","").replace(" ",""):
                                    vegas_line = float(outcome.get('point', 0))
                                    if outcome.get('name', '').lower() == 'over':
                                        over_odds = outcome.get('price')
                                    else:
                                        under_odds = outcome.get('price')
                                    line_source = f"🎯 {selected_book}"
                                    real_line_found = True
            
            # Loop 2 Fallback: If chosen bookmaker doesn't host the line, grab consensus lines on the payload
            if not real_line_found:
                for bookmaker in bookmakers_payload:
                    for market in bookmaker.get('markets', []):
                        if market.get('key', '') == target_market_key:
                            for outcome in market.get('outcomes', []):
                                api_player = outcome.get('description', '')
                                if p_name.lower().replace("-","").replace(" ","") in api_player.lower().replace("-","").replace(" ",""):
                                    vegas_line = float(outcome.get('point', 0))
                                    if outcome.get('name', '').lower() == 'over':
                                        over_odds = outcome.get('price')
                                    else:
                                        under_odds = outcome.get('price')
                                    line_source = f"⚠️ Fallback ({bookmaker.get('title')})"
                                    real_line_found = True
                                    break
            
            if not real_line_found or vegas_line is None or vegas_line == 0:
                season_median = sub_df[stat].median()
                recent_median = recent_logs[stat].median()
                vegas_line = (season_median * 0.4) + (recent_median * 0.6)
                vegas_line = round(vegas_line * 2) / 2 if stat == "3P" else int(vegas_line) + 0.5
                over_odds, under_odds = -110, -110
                line_source = f"🤖 Model Baseline"
            
            matchup_def_modifier = DEFENSIVE_YIELD_MULTIPLIER.get(stat, {}).get(opp_team, 1.0)
            
            sub_df['RATIO_FGA_MIN'] = sub_df['FGA'] / sub_df['MP'].replace(0, np.nan)
            sub_df['SEASON_MEDIAN_BASE'] = sub_df[stat].median()
            sub_df['SERIES_MIN_FACTOR'] = series_min_avg
            sub_df['SERIES_STAT_FACTOR'] = series_stat_avg
            
            features = ['RATIO_FGA_MIN', 'SERIES_MIN_FACTOR', 'SERIES_STAT_FACTOR', 'SEASON_MEDIAN_BASE']
            X = sub_df[features].fillna(0)
            y = sub_df[stat]
            
            model = xgb.XGBRegressor(n_estimators=60, learning_rate=0.1, max_depth=3, objective='reg:squarederror')
            model.fit(X, y)
            
            spread_modifier = 1.0 - (p_spread * 0.005) if stat in ["PTS", "FGA"] else 1.0
            baseline_min = sub_df['MP'].median() * total_modifier * pace_modifier
            baseline_fga_ratio = (sub_df['FGA'].median() / sub_df['MP'].median()) * spread_modifier
            
            input_data = pd.DataFrame([[baseline_fga_ratio, series_min_avg, series_stat_avg, sub_df['SEASON_MEDIAN_BASE'].iloc[0]]], columns=features)
            prediction = model.predict(input_data)[0]
            
            if stat in ["PTS", "TRB", "AST"]:
                prediction = prediction * ((total_modifier + spread_modifier) / 2)
            elif stat == "3P":
                prediction = prediction * total_modifier
                
            prediction = prediction * matchup_def_modifier
            if prediction < 0: prediction = 0.0
            
            edge_calculation = prediction - vegas_line
            rec, model_calculated_prob = generate_sharp_recommendation(prediction, vegas_line, stat)
            rationale_text = generate_rationale(p_name, stat, prediction, vegas_line, model_calculated_prob, total_modifier, matchup_def_modifier, line_source)
            
            results_list.append({
                "Player": p_name, "Team": p_team, "Prop Category": stat, "Book Line": vegas_line, "Line Source": line_source,
                "Over Odds": f"+{over_odds}" if (over_odds is not None and over_odds > 0) else str(over_odds),
                "Under Odds": f"+{under_odds}" if (under_odds is not None and under_odds > 0) else str(under_odds),
                "XGBoost Proj": round(prediction, 2), "Edge vs Book": round(edge_calculation, 2), "Model Win %": round(model_calculated_prob, 1),
                "RECOMMENDATION": rec, "Analysis Rationale": rationale_text
            })
        progress_bar.progress(min((idx + 1) / player_count, 1.0))

progress_bar.empty()

# --- DISPLAY OUTPUT ARRAYS ---
if results_list:
    board_df = pd.DataFrame(results_list)
    board_df = board_df.sort_values(by="Edge vs Book", key=abs, ascending=False).head(20)
    
    st.subheader(f"🔥 Top 20 Best Value Props ({selected_book.upper()} Focus): {selected_matchup}")
    st.dataframe(
        board_df[[
            "Player", "Team", "Prop Category", "Book Line", "Line Source", 
            "Over Odds", "Under Odds", "XGBoost Proj", "Model Win %", "RECOMMENDATION"
        ]], 
        use_container_width=True, 
        column_config={
            "XGBoost Proj": st.column_config.NumberColumn(format="%.2f 🎯"),
            "Model Win %": st.column_config.NumberColumn(format="%.1f%% 📊"),
            "Line Source": st.column_config.TextColumn("Data Source Label")
        },
        hide_index=True
    )
    
    st.subheader("📋 Contextual Analytical Deep-Dive")
    for idx, row in board_df.iterrows():
        with st.expander(f"{row['RECOMMENDATION']} - {row['Player']} ({row['Team']}) {row['Prop Category']} vs Line: {row['Book Line']}"):
            st.write(f"**Projections Breakdown:** Target Proj: **{row['XGBoost Proj']}** | Book Line: **{row['Book Line']}** (Source: **{row['Line Source']}**)")
            st.write(f"**Sharp Distribution Metrics:** Model Calculated Win Rate: **{row['Model Win %']}%** | Market Price Matrix: Over **{row['Over Odds']}** / Under **{row['Under Odds']}**")
            st.info(row['Analysis Rationale'])
else:
    st.info("No sufficient historical logs found to run deep projections for the chosen teams.")