import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import kagglehub
import requests
import sqlite3
from datetime import date, timedelta

# --- API & AUTHENTICATION CONFIGURATION ---
os.environ["KAGGLE_API_TOKEN"] = "KGAT_fcd29c0949fb0390857196aa063edb82"
BDL_API_TOKEN = "04e96a11-3aab-4146-8b3b-21b347299190"
DB_FILE = "props_history.db"

TEAM_MAPPING = {
    "BRK": "BKN",
    "PHO": "PHX",
    "CHO": "CHA",
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
st.set_page_config(page_title="NBA Daily Live Prop Predictor", layout="wide")
st.title("🏀 Reactive XGBoost Live Schedule Prop Board & Recommendation Engine")

# --- 7-DAY INTERACTIVE CALENDAR ROW ---
st.subheader("🗓️ Select Analysis Date")

if 'selected_date' not in st.session_state:
    st.session_state.selected_date = date.today()

cal_cols = st.columns(7)
for i in range(7):
    target_date = date.today() + timedelta(days=i)
    
    if i == 0:
        btn_label = f"🔥 TODAY\n({target_date.strftime('%m-%d')})"
    else:
        btn_label = f"📅 {target_date.strftime('%a')}\n({target_date.strftime('%m-%d')})"
        
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
            matchups = []
            for game in games_data:
                away = game['visitor_team']['abbreviation'].upper()
                home = game['home_team']['abbreviation'].upper()
                
                away = TEAM_MAPPING.get(away, away)
                home = TEAM_MAPPING.get(home, home)
                
                matchups.append(f"{away} @ {home}")
            return matchups
        else:
            return []
    except Exception as e:
        return []

live_matchups = get_live_schedule_for_date(BDL_API_TOKEN, st.session_state.selected_date)

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

# --- HELPER LOGIC FUNCTIONS ---
def clean_minutes(val):
    if isinstance(val, str) and ':' in val:
        parts = val.split(':')
        return float(parts[0]) + (float(parts[1]) / 60)
    return pd.to_numeric(val, errors='coerce')

def generate_betting_recommendation(edge_val, stat):
    over_threshold = 1.0 if stat != "3P" else 0.3
    under_threshold = -1.0 if stat != "3P" else -0.3
    if edge_val >= over_threshold: return "🟢 TAKE OVER"
    elif edge_val <= under_threshold: return "🔴 TAKE UNDER"
    return "⚖️ HOLD (Efficient)"

def generate_rationale(name, stat, proj, line, edge, total_mod, spread_mod):
    direction = "OVER" if edge > 0 else "UNDER"
    env_impact = []
    if total_mod > 1.05 and stat in ["PTS", "AST"]: env_impact.append("elevated game pace environment")
    elif total_mod < 0.95: env_impact.append("projected low-scoring game environment")
    if spread_mod > 1.02 and direction == "OVER": env_impact.append("increased scoring demand down the stretch")
    elif spread_mod < 0.98 and direction == "UNDER": env_impact.append("potential blowout rotation limitation risk")
    
    env_str = f" joined by {', '.join(env_impact)}" if env_impact else ""
    return f"XGBoost model detects a {abs(edge):.1f} unit variance vs book line based on running seasonal trends{env_str}. Mathematical models favor the {direction}."

# --- INTERACTIVE INTERFACE ---
st.sidebar.header("🗓️ Select Matchup")
selected_matchup = st.sidebar.selectbox("Games on Selected Day", live_matchups)
away_team, home_team = selected_matchup.split(" @ ")

# --- SIDEBAR - SPORTSBOOK ENVIRONMENT SLIDERS WITH STEP CONTROLS ---
st.sidebar.header("📊 Environmental Modifiers")
BASE_TOTAL = 218.0

if 'total_ou_val' not in st.session_state:
    st.session_state.total_ou_val = 218.0
if 'away_spread_val' not in st.session_state:
    st.session_state.away_spread_val = 0.0

# CONTROL BLOCK 1: OVER/UNDER LINE WITH PRECISION INCREMENTS
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

# CONTROL BLOCK 2: SPREAD LINE WITH PRECISION INCREMENTS
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

# --- FILTER ACTIVE ROSTERS ---
game_players_df = raw_df[raw_df['TM'].isin([away_team, home_team])].copy()
unique_players = game_players_df[['PLAYER', 'TM']].drop_duplicates()

# --- MODEL PROCESSING ENGINE ---
results_list = []
total_modifier = GAME_TOTAL_OU / BASE_TOTAL
stats_to_analyze = ["PTS", "TRB", "AST", "3P"]

progress_bar = st.progress(0)
player_count = len(unique_players)

if player_count == 0:
    st.error(f"Could not find historical matching player tags for {away_team} or {home_team} in your dataset columns.")
else:
    for idx, (_, p_row) in enumerate(unique_players.iterrows()):
        p_name = p_row['PLAYER']
        p_team = p_row['TM']
        p_spread = AWAY_SPREAD if p_team == away_team else HOME_SPREAD
        
        player_logs = raw_df[raw_df['PLAYER'] == p_name].copy()
        if len(player_logs) < 5: 
            continue
            
        player_logs['MP'] = player_logs['MP'].apply(clean_minutes)
        
        # FIX 1: ANTI-HALLUCINATION / DATA LEAKAGE PREVENTION
        player_logs = player_logs[player_logs['DATE'].dt.date < st.session_state.selected_date]
        if len(player_logs) < 5:
            continue
        
        for stat in stats_to_analyze:
            player_logs[stat] = pd.to_numeric(player_logs[stat], errors='coerce')
            player_logs['FGA'] = pd.to_numeric(player_logs['FGA'], errors='coerce')
            
            sub_df = player_logs.dropna(subset=[stat, 'MP', 'FGA']).copy()
            if len(sub_df) < 5: continue
            
            # Form calculations based on the last 5 active games
            recent_logs = sub_df.sort_values('DATE', ascending=False).head(5)
            series_min_avg = recent_logs['MP'].mean()
            series_stat_avg = recent_logs[stat].mean()
            
            # FIX 2: VEGAS-STYLE RECENCY-WEIGHTED BASELINE BLENDING (60/40)
            season_median = sub_df[stat].median()
            recent_median = recent_logs[stat].median()
            vegas_line = (season_median * 0.4) + (recent_median * 0.6)
            
            if stat == "3P": 
                vegas_line = round(vegas_line * 2) / 2
            else: 
                vegas_line = int(vegas_line) + 0.5
            
            sub_df['SERIES_MIN_FACTOR'] = series_min_avg
            sub_df['SERIES_STAT_FACTOR'] = series_stat_avg
            
            features = ['MP', 'FGA', 'SERIES_MIN_FACTOR', 'SERIES_STAT_FACTOR']
            X = sub_df[features]
            y = sub_df[stat]
            
            # Fast Ensemble Fitting Loop
            model = xgb.XGBRegressor(n_estimators=60, learning_rate=0.1, max_depth=3, objective='reg:squarederror')
            model.fit(X, y)
            
            spread_modifier = 1.0 - (p_spread * 0.005) if stat in ["PTS", "FGA"] else 1.0
            baseline_min = sub_df['MP'].median() * total_modifier
            baseline_fga = sub_df['FGA'].median() * total_modifier * spread_modifier
            
            input_data = pd.DataFrame([[
                baseline_min, baseline_fga, series_min_avg, series_stat_avg
            ]], columns=features)
            
            prediction = model.predict(input_data)[0]
            
            if stat in ["PTS", "TRB", "AST"]:
                prediction = prediction * ((total_modifier + spread_modifier) / 2)
            elif stat == "3P":
                prediction = prediction * total_modifier
                
            if prediction < 0: prediction = 0.0
            
            edge_calculation = prediction - vegas_line
            rec = generate_betting_recommendation(edge_calculation, stat)
            rationale_text = generate_rationale(p_name, stat, prediction, vegas_line, edge_calculation, total_modifier, spread_modifier)
            
            results_list.append({
                "Player": p_name,
                "Team": p_team,
                "Prop Category": stat,
                "Book Line": vegas_line,
                "XGBoost Proj": round(prediction, 2),
                "Edge vs Book": round(edge_calculation, 2),
                "RECOMMENDATION": rec,
                "Analysis Rationale": rationale_text
            })
            
        progress_bar.progress(min((idx + 1) / player_count, 1.0))

progress_bar.empty()

# --- DISPLAY OUTPUT ARRAYS ---
if results_list:
    board_df = pd.DataFrame(results_list)
    board_df = board_df.sort_values(by="Edge vs Book", key=abs, ascending=False).head(20)
    
    st.subheader(f"🔥 Top 20 Best Value Props: {selected_matchup}")
    st.dataframe(
        board_df[["Player", "Team", "Prop Category", "Book Line", "XGBoost Proj", "Edge vs Book", "RECOMMENDATION"]], 
        use_container_width=True, 
        column_config={
            "XGBoost Proj": st.column_config.NumberColumn(format="%.2f 🎯"),
            "Edge vs Book": st.column_config.NumberColumn(format="%+.2f 📈"),
        },
        hide_index=True
    )
    
    st.subheader("📋 Contextual Analytical Deep-Dive")
    for idx, row in board_df.iterrows():
        with st.expander(f"{row['RECOMMENDATION']} - {row['Player']} ({row['Team']}) {row['Prop Category']} vs Line: {row['Book Line']}"):
            st.write(f"**Projections Breakdown:** Target Proj: **{row['XGBoost Proj']}** | Book Line: **{row['Book Line']}** (Variance: **{row['Edge vs Book']:+.2f}**)")
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