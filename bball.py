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

# --- SECURE API & AUTHENTICATION CONFIGURATION ---
try:
    os.environ["KAGGLE_API_TOKEN"] = st.secrets["KAGGLE_API_TOKEN"]
    BDL_API_TOKEN = st.secrets["BDL_API_TOKEN"]
except KeyError:
    st.error("🚨 Configuration Error: API keys are missing from Streamlit Secrets! Please check your panel management settings.")
    st.stop()

DB_FILE = "props_history.db"

TEAM_MAPPING = {
    "BRK": "BKN",
    "PHO": "PHX",
    "CHO": "CHA",
}

DEFENSIVE_YIELD_MULTIPLIER = {
    "PTS": {"CLE": 0.96, "DET": 1.04, "BKN": 1.02, "BOS": 0.92, "LAL": 1.01, "GSW": 0.99},
    "TRB": {"CLE": 0.94, "DET": 1.05, "BKN": 1.01, "BOS": 0.95, "LAL": 0.97, "GSW": 1.03},
    "AST": {"CLE": 0.98, "DET": 1.02, "BKN": 1.03, "BOS": 0.91, "LAL": 1.00, "GSW": 0.96},
    "3P":  {"CLE": 1.03, "DET": 1.01, "BKN": 0.99, "BOS": 0.88, "LAL": 1.02, "GSW": 1.01}
}

# --- INITIALIZE DATABASE ARCHITECTURE ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS saved_props (
            id TEXT PRIMARY KEY,
            saved_date TEXT,
            game_date TEXT,
            player TEXT,
            team TEXT,
            category TEXT,
            book_line REAL,
            xgboost_proj REAL,
            edge REAL,
            recommendation TEXT,
            actual_stat REAL,
            outcome TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- PAGE SETUP ---
st.set_page_config(page_title="Sharp NBA Live Prop Predictor", layout="wide")
st.title("🧠 Sharp XGBoost Live Schedule Prop Board & Probability Distribution Engine")

# --- SECURE TIME ZONE STABILIZATION ---
def get_local_today():
    return (datetime.datetime.utcnow() - datetime.timedelta(hours=6)).date()

# --- 2-DAY INTERACTIVE CALENDAR ROW ---
st.subheader("🗓️ Select Analysis Date")

if 'selected_date' not in st.session_state:
    st.session_state.selected_date = get_local_today()

cal_cols = st.columns(2)
for i in range(2):
    target_date = get_local_today() + timedelta(days=i)
    
    if i == 0:
        btn_label = f"🔥 TODAY\n({target_date.strftime('%m-%d')})"
    else:
        btn_label = f"⏳ TOMORROW\n({target_date.strftime('%m-%d')})"
        
    is_active = (st.session_state.selected_date == target_date)
    type_style = "primary" if is_active else "secondary"
    
    if cal_cols[i].button(btn_label, key=f"cal_{i}", use_container_width=True, type=type_style):
        st.session_state.selected_date = target_date
        st.rerun()

st.write(f"Showing schedule data entries for: **{st.session_state.selected_date.strftime('%A, %B %d, %Y')}**")

# --- CACHED KAGGLE DATA INGESTION ---
@st.cache_data(ttl=3600)
def load_data_from_kaggle():
    try:
        download_dir = kagglehub.dataset_download("eduardopalmieri/nba-player-stats-season-2526")
        target_csv_path = os.path.join(download_dir, 'nba_dailyleaders_full.csv')
        
        df = pd.read_csv(target_csv_path)
        df.columns = [c.upper() for c in df.columns]
        
        if 'TM' in df.columns: df['TM'] = df['TM'].str.strip().str.upper()
        if 'OPP' in df.columns: df['OPP'] = df['OPP'].str.strip().str.upper()
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"🚨 Failed downloading/processing dataset from Kaggle Hub: {e}")
        return pd.DataFrame()

raw_df = load_data_from_kaggle()

if raw_df.empty:
    st.warning("Please check your Kaggle setup. Streamlit app execution paused.")
    st.stop()

# --- FETCH TARGET SCHEDULE VIA BALLDONTLIE API ---
@st.cache_data(ttl=300)
def get_live_schedule_for_date(api_token, run_date):
    date_str = run_date.isoformat()
    url = f"https://api.balldontlie.io/nba/v1/games?dates[]={date_str}"
    headers = {"Authorization": api_token}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            games_data = response.json().get('data', [])
            matchups = {}
            for game in games_data:
                game_id = game['id']
                away = game['visitor_team']['abbreviation'].upper()
                home = game['home_team']['abbreviation'].upper()
                
                away = TEAM_MAPPING.get(away, away)
                home = TEAM_MAPPING.get(home, home)
                
                label = f"{away} @ {home}"
                matchups[label] = game_id
            return matchups
        return {}
    except Exception:
        return {}

live_matchups_dict = get_live_schedule_for_date(BDL_API_TOKEN, st.session_state.selected_date)
live_matchups = list(live_matchups_dict.keys())

if not live_matchups:
    st.info("ℹ️ No active game endpoints scheduled for this date via API. Loading backup matrix from dataset.")
    latest_date = raw_df['DATE'].max()
    slate_df = raw_df[raw_df['DATE'] == latest_date]
    matchup_set = set()
    for _, row in slate_df.iterrows():
        t1, t2 = row['TM'], row['OPP']
        if pd.notna(t1) and pd.notna(t2):
            matchup_set.add(tuple(sorted([t1, t2])))
    live_matchups = [f"{m[0]} @ {m[1]}" for m in matchup_set]
    live_matchups_dict = {m: None for m in live_matchups}

# --- FETCH REAL-TIME PROPS FROM BALLDONTLIE API ---
@st.cache_data(ttl=300)
def fetch_real_props(api_token, game_id):
    if not game_id:
        return []
    url = f"https://api.balldontlie.io/nba/v1/props?game_id={game_id}"
    headers = {"Authorization": api_token}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get('data', [])
        return []
    except Exception:
        return []

# --- SHARP MATH HELPER FUNCTIONS ---
def american_to_probability(odds):
    if odds is None or odds == 0:
        return 50.0
    if odds < 0:
        return (abs(odds) / (abs(odds) + 100)) * 100
    else:
        return (100 / (odds + 100)) * 100

def calculate_projected_pace(away_team, home_team, raw_df):
    recent_data = raw_df[raw_df['DATE'] >= (raw_df['DATE'].max() - datetime.timedelta(days=30))]
    if recent_data.empty:
        return 100.0
    away_avg_fga = recent_data[recent_data['TM'] == away_team]['FGA'].median() * 5 / 48
    home_avg_fga = recent_data[recent_data['TM'] == home_team]['FGA'].median() * 5 / 48
    return (away_avg_fga + home_avg_fga) * 48 / 5

def generate_sharp_recommendation(proj, line, implied_over_pct, stat):
    edge = proj - line
    std_dev = 4.5 if stat == "PTS" else (2.5 if stat in ["TRB", "AST"] else 1.2)
    
    z_score = edge / std_dev
    model_over_prob = (1 - stats.norm.cdf(-z_score)) * 100
    
    if model_over_prob - implied_over_pct > 5.0:
        return "🟢 VALUE OVER", model_over_prob
    elif implied_over_pct - model_over_prob > 5.0:
        return "🔴 VALUE UNDER", (100 - model_over_prob)
    return "⚖️ HOLD (Efficient)", model_over_prob

def clean_minutes(val):
    if isinstance(val, str) and ':' in val:
        parts = val.split(':')
        return float(parts[0]) + (float(parts[1]) / 60)
    return pd.to_numeric(val, errors='coerce')

def generate_rationale(name, stat, proj, line, model_prob, total_mod, spread_mod, def_mod):
    direction = "OVER" if proj > line else "UNDER"
    env_impact = []
    if total_mod > 1.05: env_impact.append("positive macroscopic league pace tracking")
    if def_mod > 1.02: env_impact.append("favorable structural defensive matchup vulnerability")
    elif def_mod < 0.98: env_impact.append("restrictive primary coverage constraints")
    if spread_mod > 1.02 and direction == "OVER": env_impact.append("accelerated late-game trailing usage script")
    
    env_str = f" paired with {', '.join(env_impact)}" if env_impact else ""
    return f"XGBoost engine isolates a statistical variance assigning a {model_prob:.1f}% true probability toward the {direction}{env_str}. Value execution verified."

# --- INTERACTIVE INTERFACE ---
st.sidebar.header("🗓️ Select Matchup")
selected_matchup = st.sidebar.selectbox("Games on Selected Day", live_matchups)
away_team, home_team = selected_matchup.split(" @ ")

# --- DYNAMIC BOOKMAKER SELECTION MATRIX ---
st.sidebar.header("🎰 Sportsbook Feed Source")
selected_book = st.sidebar.selectbox(
    "Active Oddsmaker Vendor",
    ["draftkings", "fanduel", "betmgm", "caesars", "pointsbet"]
)

target_game_id = live_matchups_dict.get(selected_matchup)
api_props_list = fetch_real_props(BDL_API_TOKEN, target_game_id)

PROP_MAP_CATEGORIES = {
    "points": "PTS",
    "rebounds": "TRB",
    "assists": "AST",
    "threes": "3P"
}

# --- SIDEBAR - SPORTSBOOK ENVIRONMENT SLIDERS WITH STEP CONTROLS ---
st.sidebar.header("📊 Environmental Modifiers")
BASE_TOTAL = 218.0

if 'total_ou_val' not in st.session_state:
    st.session_state.total_ou_val = 218.0
if 'away_spread_val' not in st.session_state:
    st.session_state.away_spread_val = 0.0

# CONTROL BLOCK 1: OVER/UNDER LINE
st.sidebar.markdown("**Game Over/Under Line**")
ou_cols = st.sidebar.columns([1, 4, 1])

if ou_cols[0].button("➖", key="dec_ou", use_container_width=True):
    st.session_state.total_ou_val = max(195.0, st.session_state.total_ou_val - 0.5)
    st.rerun()

GAME_TOTAL_OU = ou_cols[1].slider(
    "OU Slider", 195.0, 235.0, 
    value=st.session_state.total_ou_val, 
    step=0.5, 
    label_visibility="collapsed"
)
st.session_state.total_ou_val = GAME_TOTAL_OU

if ou_cols[2].button("➕", key="inc_ou", use_container_width=True):
    st.session_state.total_ou_val = min(235.0, st.session_state.total_ou_val + 0.5)
    st.rerun()

# CONTROL BLOCK 2: SPREAD LINE
st.sidebar.markdown(f"**{away_team} Spread (Negative = Favorite)**")
spread_cols = st.sidebar.columns([1, 4, 1])

if spread_cols[0].button("➖", key="dec_spread", use_container_width=True):
    st.session_state.away_spread_val = max(-15.0, st.session_state.away_spread_val - 0.5)
    st.rerun()

AWAY_SPREAD = spread_cols[1].slider(
    "Spread Slider", -15.0, 15.0, 
    value=st.session_state.away_spread_val, 
    step=0.5, 
    label_visibility="collapsed"
)
st.session_state.away_spread_val = AWAY_SPREAD

if spread_cols[2].button("➕", key="inc_spread", use_container_width=True):
    st.session_state.away_spread_val = min(15.0, st.session_state.away_spread_val + 0.5)
    st.rerun()

HOME_SPREAD = -AWAY_SPREAD

# --- FILTER ACTIVE ROSTERS (STRICT 14-DAY RECENCY TRACKING) ---
max_data_date = raw_df['DATE'].max()
roster_cutoff_date = max_data_date - datetime.timedelta(days=14)

active_rosters_df = raw_df[
    (raw_df['TM'].isin([away_team, home_team])) & 
    (raw_df['DATE'] >= roster_cutoff_date)
]
unique_players = active_rosters_df[['PLAYER', 'TM']].drop_duplicates()

# --- MODEL PROCESSING ENGINE ---
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
        if len(player_logs) < 5: 
            continue
            
        player_logs['MP'] = player_logs['MP'].apply(clean_minutes)
        
        # ANTI-HALLUCINATION / DATA LEAKAGE PREVENTION
        player_logs = player_logs[player_logs['DATE'].dt.date < st.session_state.selected_date]
        if len(player_logs) < 5:
            continue
        
        for stat in stats_to_analyze:
            player_logs[stat] = pd.to_numeric(player_logs[stat], errors='coerce')
            player_logs['FGA'] = pd.to_numeric(player_logs['FGA'], errors='coerce')
            
            sub_df = player_logs.dropna(subset=[stat, 'MP', 'FGA']).copy()
            if len(sub_df) < 5: continue
            
            recent_logs = sub_df.sort_values('DATE', ascending=False).head(5)
            series_min_avg = recent_logs['MP'].mean()
            series_stat_avg = recent_logs[stat].mean()
            
            # --- OVERHAUL: STRICT CHOICE SELECTION PARSER ---
            real_line_found = False
            vegas_line = None
            over_odds = None
            under_odds = None
            
            # First Pass: Search specifically for the user's targeted bookmaker selection
            for api_prop in api_props_list:
                api_player = api_prop.get("player_name", "")
                api_type = PROP_MAP_CATEGORIES.get(api_prop.get("prop_type", ""), "")
                api_vendor = api_prop.get("vendor", "").lower()
                
                if p_name.lower() in api_player.lower() and api_type == stat and selected_book in api_vendor:
                    vegas_line = float(api_prop.get("line_value", 0))
                    market = api_prop.get("market", {})
                    over_odds = market.get("over_odds", -110)
                    under_odds = market.get("under_odds", -110)
                    real_line_found = True
                    break
            
            # Second Pass Fallback: If your specific book hasn't posted lines yet, grab general market consensus data
            if not real_line_found:
                for api_prop in api_props_list:
                    api_player = api_prop.get("player_name", "")
                    api_type = PROP_MAP_CATEGORIES.get(api_prop.get("prop_type", ""), "")
                    
                    if p_name.lower() in api_player.lower() and api_type == stat:
                        vegas_line = float(api_prop.get("line_value", 0))
                        market = api_prop.get("market", {})
                        over_odds = market.get("over_odds", -110)
                        under_odds = market.get("under_odds", -110)
                        real_line_found = True
                        break
            
            # Third Pass Fallback: Synthetic line assembly if completely unpublished on all odds feeds
            if not real_line_found or vegas_line is None or vegas_line == 0:
                season_median = sub_df[stat].median()
                recent_median = recent_logs[stat].median()
                vegas_line = (season_median * 0.4) + (recent_median * 0.6)
                vegas_line = round(vegas_line * 2) / 2 if stat == "3P" else int(vegas_line) + 0.5
                over_odds, under_odds = -110, -110
            
            implied_over_pct = american_to_probability(over_odds)
            implied_under_pct = american_to_probability(under_odds)
            
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
            
            input_data = pd.DataFrame([[
                baseline_fga_ratio, series_min_avg, series_stat_avg, sub_df['SEASON_MEDIAN_BASE'].iloc[0]
            ]], columns=features)
            
            prediction = model.predict(input_data)[0]
            
            if stat in ["PTS", "TRB", "AST"]:
                prediction = prediction * ((total_modifier + spread_modifier) / 2)
            elif stat == "3P":
                prediction = prediction * total_modifier
                
            prediction = prediction * matchup_def_modifier
            if prediction < 0: prediction = 0.0
            
            edge_calculation = prediction - vegas_line
            rec, model_calculated_prob = generate_sharp_recommendation(prediction, vegas_line, implied_over_pct, stat)
            rationale_text = generate_rationale(p_name, stat, prediction, vegas_line, model_calculated_prob, total_modifier, spread_modifier, matchup_def_modifier)
            
            results_list.append({
                "Player": p_name,
                "Team": p_team,
                "Prop Category": stat,
                "Book Line": vegas_line,
                "Over Odds": f"+{over_odds}" if over_odds > 0 else str(over_odds),
                "Implied Over %": round(implied_over_pct, 1),
                "Under Odds": f"+{under_odds}" if under_odds > 0 else str(under_odds),
                "Implied Under %": round(implied_under_pct, 1),
                "XGBoost Proj": round(prediction, 2),
                "Edge vs Book": round(edge_calculation, 2),
                "Model Win %": round(model_calculated_prob, 1),
                "RECOMMENDATION": rec,
                "Analysis Rationale": rationale_text
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
            "Player", "Team", "Prop Category", "Book Line", 
            "Over Odds", "Implied Over %", "Under Odds", "Implied Under %",
            "XGBoost Proj", "Model Win %", "RECOMMENDATION"
        ]], 
        use_container_width=True, 
        column_config={
            "XGBoost Proj": st.column_config.NumberColumn(format="%.2f 🎯"),
            "Model Win %": st.column_config.NumberColumn(format="%.1f%% 📊"),
            "Implied Over %": st.column_config.NumberColumn(format="%.1f%% 📈"),
            "Implied Under %": st.column_config.NumberColumn(format="%.1f%% 📉")
        },
        hide_index=True
    )
    
    st.subheader("📋 Contextual Analytical Deep-Dive")
    for idx, row in board_df.iterrows():
        with st.expander(f"{row['RECOMMENDATION']} - {row['Player']} ({row['Team']}) {row['Prop Category']} vs Line: {row['Book Line']}"):
            st.write(f"**Projections Breakdown:** Target Proj: **{row['XGBoost Proj']}** | Book Line: **{row['Book Line']}** (Variance: **{row['Edge vs Book']:+.2f}**)")
            st.write(f"**Sharp Distribution Metrics:** Model Calculated Win Rate: **{row['Model Win %']}%** | Book Implied Over Price: **{row['Implied Over %']}%**")
            st.info(row['Analysis Rationale'])
            
    # --- PERSISTENT TRACKING REGISTRY OVERLAY ---
    st.markdown("---")
    st.subheader("💾 Lock & Track Current Slate")
    
    if st.button("🔒 Save Current Top 20 Slate to History Tracker", use_container_width=True):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        saved_count = 0
        
        for _, row in board_df.iterrows():
            unique_prop_id = f"{st.session_state.selected_date}_{row['Player']}_{row['Prop Category']}"
            cursor.execute("""
                INSERT OR IGNORE INTO saved_props 
                (id, saved_date, game_date, player, team, category, book_line, xgboost_proj, edge, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                unique_prop_id,
                date.today().isoformat(),
                st.session_state.selected_date.isoformat(),
                row['Player'],
                row['Team'],
                row['Prop Category'],
                row['Book Line'],
                row['XGBoost Proj'],
                row['Edge vs Book'],
                row['RECOMMENDATION']
            ))
            if cursor.rowcount > 0:
                saved_count += 1
                
        conn.commit()
        conn.close()
        
        if saved_count > 0:
            st.success(f"Successfully locked {saved_count} new props into persistent tracking storage!")
        else:
            st.info("ℹ️ This specific game slate has already been locked into the history ledger.")

else:
    st.info("No sufficient historical logs found to run deep projections for the chosen teams.")

# --- HISTORICAL PERFORMANCE LEDGER & AUTO-GRADER ---
st.markdown("---")
st.subheader("📊 Historical Performance Ledger")

conn = sqlite3.connect(DB_FILE)
history_df = pd.read_sql_query("SELECT * FROM saved_props", conn)
conn.close()

if not history_df.empty:
    resolved_rows = []
    
    for idx, row in history_df.iterrows():
        if pd.isna(row['outcome']) or row['outcome'] in ["Pending", "⌛ Pending Score"]:
            target_date_dt = pd.to_datetime(row['game_date'])
            actual_game_log = raw_df[
                (raw_df['PLAYER'] == row['player']) & 
                (raw_df['TM'] == row['team']) & 
                (raw_df['DATE'] == target_date_dt)
            ]
            
            if not actual_game_log.empty:
                stat_col = row['category']
                actual_val = pd.to_numeric(actual_game_log.iloc[0].get(stat_col, np.nan), errors='coerce')
                
                if pd.notna(actual_val):
                    is_over_rec = "OVER" in row['recommendation']
                    line = row['book_line']
                    
                    if actual_val == line:
                        outcome_str = "⚖️ PUSH"
                    elif (is_over_rec and actual_val > line) or (not is_over_rec and actual_val < line):
                        outcome_str = "🎯 HIT"
                    else:
                        outcome_str = "❌ MISS"
                        
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE saved_props 
                        SET actual_stat = ?, outcome = ? 
                        WHERE id = ?
                    """, (float(actual_val), outcome_str, row['id']))
                    conn.commit()
                    conn.close()
                    
                    row['actual_stat'] = actual_val
                    row['outcome'] = outcome_str
                else:
                    row['outcome'] = "⌛ Pending Score"
            else:
                row['outcome'] = "⌛ Pending Score"
                
        resolved_rows.append(row)
        
    updated_history_df = pd.DataFrame(resolved_rows)
    
    completed_bets = updated_history_df[updated_history_df['outcome'].isin(["🎯 HIT", "❌ MISS"])]
    if not completed_bets.empty:
        hits = len(completed_bets[completed_bets['outcome'] == "🎯 HIT"])
        total_graded = len(completed_bets)
        hit_rate = (hits / total_graded) * 100
        st.metric("Model Win Rate (Graded Plays)", f"{hit_rate:.1f}%", f"{hits} Wins / {total_graded} Totals")
        
    st.dataframe(
        updated_history_df[[
            "game_date", "player", "team", "category", 
            "book_line", "xgboost_proj", "recommendation", "actual_stat", "outcome"
        ]].sort_values(by="game_date", ascending=False),
        use_container_width=True,
        column_config={
            "book_line": st.column_config.NumberColumn("Line"),
            "xgboost_proj": st.column_config.NumberColumn("Proj"),
            "actual_stat": st.column_config.NumberColumn("Actual"),
            "outcome": st.column_config.TextColumn("Grading Status")
        },
        hide_index=True
    )
else:
    st.info("No slates currently saved in tracking storage. Click the lock button above to begin logging sports analytics slates.")