import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from xgboost import XGBRegressor
# Backend functions from predictor.py
from predictor import predict_team, get_actual_dream_team, get_model_artifact, select_team_ilp




# load_test_data is intentionally defined AFTER load_full_dataset (below)
# so it can reuse the cached full dataset. The call is deferred to after both
# cached loaders are defined. See the call site below the fixture loader.

@st.cache_data
def load_fixture_schedule(dataset_path="cleaned_dataset.csv"):
    """
    Loads just the scheduling columns from the dataset and caches them 
    so the UI doesn't lag when changing dates.
    """
    # Only read the columns we need for the UI dropdowns
    df = pd.read_csv(dataset_path, usecols=['date', 'team', 'opposition','match_type'], low_memory=False)
    # Ensure date is just a date object (no time) for exact matching with Streamlit's date_input
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

# Load the schedule into memory
fixtures_df = load_fixture_schedule()

# Secure Sidebar Control Panel 
with st.sidebar:
    st.markdown("<h2 style='text-align: center; margin-bottom: 30px;'>⚡ CONTROL PANEL</h2>", unsafe_allow_html=True)
    
    # AI Configuration
    st.markdown("<h4 style='color: #00f0ff; font-size: 16px;'>1. NEURAL ENGINE API</h4>", unsafe_allow_html=True)
    existing_key = os.environ.get("GROQ_API_KEY", "")
    groq_key = st.text_input("Groq API Key", value=existing_key, type="password", help="Required to unlock AI Audio & Explainability")
    
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("API Key Loaded!")
        
    # Advanced Settings
    st.markdown("<hr style='border-color: #1e293b; margin: 25px 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00f0ff; font-size: 16px;'>2. GENERATION PREFERENCES</h4>", unsafe_allow_html=True)
    
    # Interactive Toggle: Only enable the audio switch if they provided a key
    enable_audio = st.toggle(
        "Enable AI Audio Synthesis", 
        value=True if groq_key else False, 
        disabled=not bool(groq_key),
        help="Turn off to skip audio generation and speed up team selection."
    )
    
    # Live System Status
    st.markdown("<hr style='border-color: #1e293b; margin: 25px 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00f0ff; font-size: 16px;'>3. SYSTEM DIAGNOSTICS</h4>", unsafe_allow_html=True)
    
    # Dynamic status indicators
    db_status = "<span class='status-dot-green'>●</span> ONLINE"
    ml_status = "<span class='status-dot-green'>●</span> READY"
    api_status = "<span class='status-dot-green'>●</span> CONNECTED" if groq_key else "<span class='status-dot-red'>●</span> OFFLINE (Requires Key)"
    audio_status = "<span class='status-dot-green'>●</span> SYNTHESIS ACTIVE" if (groq_key and enable_audio) else "<span class='status-dot-red'>●</span> MUTED"
    
    st.markdown(f"""
        <div class='glass-box' style='padding: 15px;'>
            <div class='status-indicator'>CRICSHEET DB: {db_status}</div>
            <div class='status-indicator'>XGBOOST ML: {ml_status}</div>
            <div class='status-indicator'>GROQ LLM: {api_status}</div>
            <div class='status-indicator'>AUDIO ENGINE: {audio_status}</div>
        </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_full_dataset(dataset_path="cleaned_dataset.csv"):
    """
    Loads and caches the FULL dataset (all columns, all rows) exactly once.
    Reusing this avoids re-reading the CSV on every button click, which is
    critical for the 10-second team generation limit.
    """
    df = pd.read_csv(dataset_path, low_memory=False)
    df['date']     = pd.to_datetime(df['date'], errors='coerce')
    df['match_id'] = df['match_id'].astype(str)
    df = df.dropna(subset=['date'])
    return df


# Derive test_df from the single cached full dataset — no extra CSV read.
# test_df is used by the Model UI quick-evaluate path (generate_evaluation_csv).
def _build_test_df():
    df = load_full_dataset()
    return df[df['date'] > pd.Timestamp('2024-06-30')].copy()

test_df = _build_test_df()


def fetch_squad_data(team_1, team_2, match_date, dataset_path="cleaned_dataset.csv"):
    """
    Builds a pre-match squad profile for two teams.

    The problem statement says: "Toss has not happened yet — you have the complete
    squad of each team (15+ members) but NO information about actual match outcomes."

    So we must NOT return the real match row (which contains fantasy_points from the
    future).  Instead we:
      1. Find which players appeared in the actual match on that date (squad list only).
      2. For every player in the squad, reconstruct their historical feature vector
         using only data STRICTLY BEFORE the match date (i.e. the same information
         available to someone predicting on the eve of the match).
      3. Return that squad DataFrame — with historical rolling features but with
         fantasy_points = 0 (unknown pre-match), so the ML model predicts from
         legitimate inputs.

    Uses load_full_dataset() so the CSV is read only ONCE per Streamlit session,
    keeping team generation well inside the 10-second contest limit.
    """
    # Use the cached full dataset — zero disk I/O after first call
    df = load_full_dataset(dataset_path)

    target_date = pd.to_datetime(match_date)

    #  Identify the squad from the actual match row 
    # We use the match row ONLY to get the list of player names + their roles/teams.
    # We do NOT use any stat columns from it.
    match_row = df[
        (df['date'].dt.date == target_date.date()) &
        (
            ((df['team'] == team_1) & (df['opposition'] == team_2)) |
            ((df['team'] == team_2) & (df['opposition'] == team_1))
        )
    ]

    if match_row.empty:
        raise ValueError(
            f"No match found for {team_1} vs {team_2} on {match_date}. "
            "Check team names and date."
        )

    match_id   = match_row['match_id'].iloc[0]
    squad_rows = match_row[match_row['match_id'] == match_id][
        ['player', 'team', 'role']
    ].drop_duplicates(subset='player').reset_index(drop=True)

    #  For each player, take their LAST historical row before match_date ─
    # This is the feature vector that was legitimately available before the match.
    historical = df[df['date'] < target_date].copy()

    squad_records = []
    for _, player_info in squad_rows.iterrows():
        player_name = player_info['player']
        player_hist = historical[historical['player'] == player_name]

        if player_hist.empty:
            # New/unknown player — create a zero-feature row so the model still runs
            feature_row              = pd.Series(0, index=df.columns)
            feature_row['player']    = player_name
            feature_row['team']      = player_info['team']
            feature_row['role']      = player_info['role']
            feature_row['fantasy_points'] = 0.0
        else:
            # Take the most recent historical appearance as the feature snapshot
            feature_row           = player_hist.sort_values('date').iloc[-1].copy()
            # Overwrite role/team from the squad row (might have changed)
            feature_row['team']   = player_info['team']
            feature_row['role']   = player_info['role']
            # Zero out future-only columns so the model never sees actual match stats
            feature_row['fantasy_points'] = 0.0

        squad_records.append(feature_row)

    squad_df = pd.DataFrame(squad_records).reset_index(drop=True)
    return squad_df
   


def generate_evaluation_csv(eval_df, start_date, end_date):
    """
    Evaluates the ProductUI_Model over a specified date range and returns a
    DataFrame matching the strict Inter IIT Tech Meet 13.0 CSV schema.

    Column contract (identical to the inline Model UI loop so both paths
    produce the same output):
        Match Date | Team 1 | Team 2
        Predicted Player 1..11  | Predicted Player 1..11 Points  <- model predicted_fp
        Dream Team Player 1..11 | Dream Team Player 1..11 Points <- actual fantasy_points
        Total Predicted Points | Total Dream Team Points | MAE
    """
    mask = (
        (eval_df['date'] >= pd.Timestamp(start_date)) &
        (eval_df['date'] <= pd.Timestamp(end_date))
    )
    filtered = eval_df[mask]

    match_ids = filtered['match_id'].unique()
    results   = []

    for match_id in match_ids:
        match_data = (
            filtered[filtered['match_id'] == match_id]
            .copy().reset_index(drop=True)
        )

        if len(match_data) < 11:
            continue

        match_date_str = match_data['date'].iloc[0].strftime('%Y-%m-%d')
        team_a = match_data['team'].iloc[0]
        team_b = (
            match_data['opposition'].iloc[0]
            if 'opposition' in match_data.columns else "Unknown"
        )

        # Predictions from the saved ProductUI_Model
        pred_team , _  = predict_team(match_data)          # has predicted_fp column
        actual_team = get_actual_dream_team(match_data) # predicted_fp = actual fantasy_points

        # Actual fantasy points keyed by player (used for pred_total calculation)
        fp_map      = match_data.set_index('player')['fantasy_points'].to_dict()
        pred_total  = sum(fp_map.get(p, 0) for p in pred_team['player'])
        dream_total = actual_team['predicted_fp'].sum()
        mae         = abs(dream_total - pred_total)

        row = {
            'Match Date': match_date_str,
            'Team 1':     team_a,
            'Team 2':     team_b,
        }

        # Predicted 11: store the MODEL'S predicted_fp, sorted descending
        pred_sorted = pred_team.sort_values('predicted_fp', ascending=False)
        for i in range(11):
            pname = pred_sorted['player'].iloc[i]            if i < len(pred_sorted) else "N/A"
            ppts  = float(pred_sorted['predicted_fp'].iloc[i]) if i < len(pred_sorted) else 0.0
            row[f'Predicted Player {i+1}']        = pname
            row[f'Predicted Player {i+1} Points'] = round(ppts, 2)

        # Dream Team 11: store the REAL fantasy_points, sorted descending
        actual_sorted = actual_team.sort_values('predicted_fp', ascending=False)
        for i in range(11):
            aname = actual_sorted['player'].iloc[i]             if i < len(actual_sorted) else "N/A"
            apts  = float(actual_sorted['predicted_fp'].iloc[i]) if i < len(actual_sorted) else 0.0
            row[f'Dream Team Player {i+1}']        = aname
            row[f'Dream Team Player {i+1} Points'] = round(apts, 2)

        row['Total Predicted Points']  = round(pred_total,  2)
        row['Total Dream Team Points'] = round(dream_total, 2)
        row['MAE']                     = round(mae, 2)

        results.append(row)

    return pd.DataFrame(results)



# Page Configuration 
st.set_page_config(
    page_title="Dream11 Next-Gen Team Builder",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import plotly.express as px
import plotly.graph_objects as go


#Cyberpunk Theme CSS Injection
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&display=swap');
    
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
        font-family: 'Rajdhani', sans-serif;
    }
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #00f0ff !important;
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 2px;
    }
    .status-indicator {
        font-size: 15px;
        color: #94a3b8;
        margin-bottom: 12px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .status-dot-green {
        color: #10b981;
        text-shadow: 0 0 8px #10b981;
        margin-right: 8px;
    }
    .status-dot-red {
        color: #ef4444;
        text-shadow: 0 0 8px #ef4444;
        margin-right: 8px;
    }
            


    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif !important;
        text-transform: uppercase;
        color: #ffffff !important;
    }
    .neon-title {
        color: #00f0ff;
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.5);
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0px;
    }
    /* Glassmorphism Containers */
    .glass-box {
        background: rgba(16, 24, 39, 0.7);
        border: 1px solid #1e293b;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    /* Player Cards */
    .player-card {
        background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px 10px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .player-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 15px rgba(0, 240, 255, 0.4);
        border-color: #00f0ff;
    }
    .role-header {
        text-align: center;
        color: #94a3b8;
        font-size: 14px;
        letter-spacing: 3px;
        margin: 25px 0 15px 0;
    }
    /* Neon Buttons */
    .stButton>button {
        background: transparent !important;
        color: #00f0ff !important;
        border: 1px solid #00f0ff !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px rgba(0, 240, 255, 0.2) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background: #00f0ff !important;
        color: #000000 !important;
        box-shadow: 0 0 20px rgba(0, 240, 255, 0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Top Header 
st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <div class='neon-title'>NEXT-GEN TEAM BUILDER</div>
        <p style='color: #94a3b8; letter-spacing: 2px;'>AI-POWERED FANTASY CRICKET SELECTION </p>
    </div>
""", unsafe_allow_html=True)

# Top Dashboard Brand Header 


# Create the two required interfaces using tabs



def create_player_html(player_name, points, role, conf, team, is_cap=False, is_vc=False):
    """Generates the HTML for a single neon player card."""
    # Color code the confidence levels
    conf_color = "#00f0ff" if conf == "High" else ("#f59e0b" if conf == "Medium" else "#ef4444")
    
    # Format name (e.g., "Wanindu Hasaranga" -> "W. Hasaranga")
    name_parts = player_name.split()
    short_name = f"{name_parts[0][0]}. {' '.join(name_parts[1:])}" if len(name_parts) > 1 else player_name
    

    # Cap/VC specific styling
    border_color = conf_color
    box_shadow = ""
    badge_html = ""
    
    if is_cap:
        border_color = "#fbbf24" # Neon Gold for Captain
        box_shadow = "box-shadow: 0 0 15px rgba(251, 191, 36, 0.4);"
        badge_html = "<div style='position: absolute; top: -12px; right: -12px; background: #fbbf24; color: #000; border-radius: 50%; width: 28px; height: 28px; line-height: 28px; text-align: center; font-weight: 900; font-size: 14px; box-shadow: 0 0 10px #fbbf24; z-index: 10;'>C</div>"
    elif is_vc:
        border_color = "#cbd5e1" # Neon Silver for VC
        box_shadow = "box-shadow: 0 0 15px rgba(203, 213, 225, 0.3);"
        badge_html = "<div style='position: absolute; top: -12px; right: -12px; background: #cbd5e1; color: #000; border-radius: 50%; width: 28px; height: 28px; line-height: 28px; text-align: center; font-weight: 900; font-size: 12px; box-shadow: 0 0 10px #cbd5e1; z-index: 10;'>VC</div>"
    
    html_string = f"<div class='player-card' style='border-top: 3px solid {border_color}; {box_shadow} position: relative; margin-top: 15px;'>{badge_html}<div style='font-size: 11px; color: #64748b; font-weight: bold;'>{team[:3].upper()}</div><div style='font-size: 15px; font-weight: 700; color: #f8fafc; margin: 8px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;'>{short_name}</div><div style='font-size: 22px; font-weight: 700; color: {border_color};'>{points:.1f}</div><div style='font-size: 10px; background: rgba(0,0,0,0.3); border-radius: 4px; padding: 3px; margin-top: 8px; color: {conf_color};'>{conf.upper()} CONFIDENCE</div></div>"
    
    return html_string

tab1, tab2 = st.tabs(["🎯 PRODUCT UI: TEAM BUILDER", "📊 MODEL UI: PERFORMANCE METRICS"])


# INTERFACE 1: PRODUCT UI

with tab1:
   
    # Match Setup Container 
    

    with st.container(border=True):
        st.markdown("<h4 style='color: #00f0ff; margin-top: 0; margin-bottom: 10px;'>⚙️ MATCH SETUP</h4>", unsafe_allow_html=True)


  
        col_date, col_type = st.columns(2)
        col_t1, col_t2 = st.columns(2)
   
    
        with col_date:
            match_date = st.date_input("MATCH DATE", value=datetime.date(2024, 7, 1))


        if match_date:
            daily_matches = fixtures_df[fixtures_df['date'] == match_date]
            available_match_types = daily_matches['match_type'].dropna().unique().tolist()
        else:
            daily_matches = pd.DataFrame()
            available_match_types = []
        
    

    # Filter by Match Type 
        with col_type:
            if not available_match_types:
                match_type = st.selectbox("MATCH TYPE", ["N/A"], disabled=True)
                type_filtered_matches = pd.DataFrame()
            else:
                match_type = st.selectbox("MATCH TYPE", available_match_types)
            # Filter matches down to ONLY the selected format
                type_filtered_matches = daily_matches[daily_matches['match_type'] == match_type]

    #  Filter Teams based on the selected Format 
        available_teams = type_filtered_matches['team'].unique().tolist() if not type_filtered_matches.empty else []
    
        with col_t1:
            if not available_teams:
                team_1 = st.selectbox("TEAM 1", ["No Teams Available"], disabled=True)
            else:
                team_1 = st.selectbox("TEAM 1", available_teams, index=None, placeholder="Select Team 1...")
            
        with col_t2:
            if not available_teams or team_1 is None:
                team_2 = st.selectbox("TEAM 2", ["No Teams Available"], disabled=True)
            else:
                # Only show opponents that actually played team_1 on this date
                valid_opponents = (
                    daily_matches[daily_matches['team'] == team_1]['opposition']
                    .unique().tolist()
                )
                if not valid_opponents:
                    team_2 = st.selectbox("TEAM 2", ["No opponents found"], disabled=True)
                else:
                    team_2 = st.selectbox(
                        "TEAM 2", valid_opponents,
                        placeholder="Select Opponent..."
                    )
            
    
        
    st.markdown("</div>", unsafe_allow_html=True)


    
    
    # 1. Ask for the Match Date FIRST, because the dropdowns depend on it
    #match_date = st.date_input("Match Date", min_value=datetime.date(2024, 7, 1))
    
    # 2. Filter the schedule for the chosen date
    #daily_matches = fixtures_df[fixtures_df['date'] == match_date]
    #available_teams = daily_matches['team'].unique().tolist()
    
    #col1, col2 = st.columns(2)



    if st.button("Generate Dream Team", type="primary"):
        if team_1 and team_2:
            with st.spinner('AI is analyzing player stats and match conditions...'):
                try:
                   
                    squad_df = fetch_squad_data(team_1, team_2, match_date)
                    
                    # Mocking the dataframe for UI demonstration purposes
                    # Run the prediction and trigger AI if the API key is provided

                    pred_team, audio_path = predict_team(squad_df, team1=team_1, team2=team_2, generate_ai = enable_audio)
                    
                    st.success("Team Generated Successfully!")

                    # Top Summary Banner 
                    # base_pts  = raw model predictions summed (shown in the table)
                    # total_pts = with C×2 and VC×1.5 applied (your actual Dream11 score)
                    base_pts   = pred_team['predicted_fp'].sum()
                    total_pts  = pred_team['display_fp'].sum()
                    high_conf  = len(pred_team[pred_team['confidence'] == 'High'])
                    role_counts = pred_team['role'].value_counts()
                    composition_str = f"{role_counts.get('BAT', 0)} BAT | {role_counts.get('AR', 0)} AR | {role_counts.get('BOWL', 0)} BOWL | {role_counts.get('WK', 0)} WK"

                    st.markdown(f"""
                        <div class='glass-box' style='display: flex; justify-content: space-between; text-align: center;'>
                            <div>
                                <span style='color:#94a3b8; font-size:12px;'>BASE PREDICTED FP (11 players)</span><br>
                                <span style='font-size: 24px; color: #00f0ff; font-weight:bold;'>{base_pts:.1f}</span><br>
                                <span style='color:#64748b; font-size:11px;'>No C/VC bonus</span>
                            </div>
                            <div>
                                <span style='color:#94a3b8; font-size:12px;'>DREAM11 SCORE (C×2, VC×1.5)</span><br>
                                <span style='font-size: 24px; color: #f59e0b; font-weight:bold;'>{total_pts:.1f}</span><br>
                                <span style='color:#64748b; font-size:11px;'>Your Actual Contest Points</span>
                            </div>
                            <div>
                                <span style='color:#94a3b8; font-size:12px;'>HIGH CONFIDENCE PICKS</span><br>
                                <span style='font-size: 24px; color: #f8fafc; font-weight:bold;'>{high_conf}/11</span>
                            </div>
                            <div>
                                <span style='color:#94a3b8; font-size:12px;'>SQUAD COMPOSITION</span><br>
                                <span style='font-size: 24px; color: #00f0ff; font-weight:bold;'>{composition_str}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)



                    
                    
                    # The Pitch Formation (Visual Cards) 
                    st.markdown("<h3 style='text-align: center; margin-top: 30px; color: #00f0ff;'>PREDICTED PLAYING XI</h3>", unsafe_allow_html=True)
                    
                    # Order of roles on the pitch
                    display_order = ['WK', 'BAT', 'AR', 'BOWL']
                    role_titles = {'WK': 'WICKET-KEEPER', 'BAT': 'BATSMEN', 'AR': 'ALL-ROUNDERS', 'BOWL': 'BOWLERS'}


                    for role in display_order:
                        role_players = pred_team[pred_team['role'] == role]
                        if not role_players.empty:
                            st.markdown(f"<div class='role-header'>— {role_titles[role]} —</div>", unsafe_allow_html=True)
                            # Create dynamic columns based on number of players in this role
                            cols = st.columns(len(role_players))
                            for idx, (_, player_row) in enumerate(role_players.iterrows()):
                                with cols[idx]:

                                    # Identify C and VC dynamically from the dataframe
                                    is_cap = bool(player_row.get('captain', False))
                                    is_vc = bool(player_row.get('vc', False))
                                    st.markdown(
                                        create_player_html(
                                            player_row['player'], 
                                            player_row['display_fp'], 
                                            player_row['role'], 
                                            player_row['confidence'],
                                            player_row['team'],
                                            is_cap = is_cap,
                                            is_vc = is_vc
                                        ), 
                                        unsafe_allow_html=True
                                    )
                                    
                    
                    # Detailed View Dataframe (Custom Cyberpunk Table)
                    st.markdown("<h3 style='margin-top: 40px; color: #00f0ff; text-align: center; letter-spacing: 2px;'>ALL 11 PLAYERS - DETAILED VIEW</h3>", unsafe_allow_html=True)
                    
                    # Sort players by predicted points so the best are at the top
                    display_df = pred_team.sort_values(by='predicted_fp', ascending=False)
                    
                   
                    

                 

                    # Completely flattened CSS and Table Header
                    table_html = """<style>.cyber-table { width: 100%; border-collapse: collapse; margin: 20px 0; font-family: 'Rajdhani', sans-serif; font-size: 16px; color: #e2e8f0; background: rgba(16, 24, 39, 0.6); border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); } .cyber-table thead tr { background-color: #111827; color: #00f0ff; text-align: left; border-bottom: 2px solid #00f0ff; } .cyber-table th, .cyber-table td { padding: 14px 20px; } .cyber-table tbody tr { border-bottom: 1px solid #1e293b; transition: all 0.2s ease; } .cyber-table tbody tr:hover { background-color: rgba(0, 240, 255, 0.05); border-left: 3px solid #00f0ff; } .cyber-table tbody tr:last-of-type { border-bottom: none; } .pill { padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 800; color: #000; text-transform: uppercase; } .pill-high { background: #00f0ff; } .pill-med { background: #f59e0b; } .pill-low { background: #ef4444; } .pill-cap { background: #fbbf24; margin-left: 10px; box-shadow: 0 0 8px rgba(251, 191, 36, 0.6);} .pill-vc { background: #cbd5e1; margin-left: 10px; box-shadow: 0 0 8px rgba(203, 213, 225, 0.6);}</style><table class="cyber-table"><thead><tr><th>PLAYER NAME</th><th>TEAM</th><th>ROLE</th><th>BASE EXPECTED FP</th><th>CONFIDENCE</th></tr></thead><tbody>"""

            



                    # Injecting DataFrame rows into HTML
                    for _, row in display_df.iterrows():
                        # Determine Confidence Pill Color
                        conf = row['confidence']
                        pill_class = "pill-high" if conf == "High" else ("pill-med" if conf == "Medium" else "pill-low")
                        conf_html = f"<span class='pill {pill_class}'>{conf}</span>"

                        # Determine C / VC Badges
                        name_html = row['player']
                        if row.get('captain', False):
                            name_html += "<span class='pill pill-cap'>C</span>"
                        elif row.get('vc', False):
                            name_html += "<span class='pill pill-vc'>VC</span>"

                        
                        # ONE SINGLE LINE: No indents, no way for Streamlit to break it.
                        table_html += f"<tr><td style='font-weight: bold; color: #f8fafc; font-size: 18px;'>{name_html}</td><td>{row['team']}</td><td style='color: #94a3b8; font-weight: 600; letter-spacing: 1px;'>{row['role']}</td><td style='color: #00f0ff; font-weight: bold; font-size: 18px;'>{row['predicted_fp']:.2f}</td><td>{conf_html}</td></tr>"

                       
                        # Add the AI Insight Commentary Row directly if it exists!
                        if 'commentary' in row and pd.notna(row['commentary']) and row['commentary'] != "":
                            table_html += f"<tr><td colspan='5' style='padding-top: 0; padding-bottom: 20px; color: #94a3b8; font-size: 14px; border-bottom: 1px solid #1e293b;'><span style='color: #00f0ff; font-weight: bold;'>🧠 AI INSIGHT:</span> <i>{row['commentary']}</i></td></tr>"

                    

                    # Close Table
                    table_html += "</tbody></table>"


                    
                    # Render Table in Streamlit
                    st.markdown(table_html, unsafe_allow_html=True)


                    
                    
                    


                   

                    # Interactive Audio Player 
                    st.markdown("<h4 style='color:#00f0ff;margin-top:40px;margin-bottom:10px;'>🧠 AI COMMENTARY & AUDIO BREAKDOWN</h4>", unsafe_allow_html=True)
                    if audio_path and os.path.exists(audio_path):
                        st.audio(audio_path)
                    elif not groq_key:
                        st.warning("⚠️ Enter your Groq API Key in the sidebar to unlock the specific AI Insights and Audio Breakdown.")


                    
                    
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter both team names.")







# INTERFACE 2: MODEL UI

with tab2:
    st.markdown(
        "<div style='text-align:center;margin-bottom:30px;margin-top:10px;'>"
        "<h2 style='color:#00f0ff;margin-bottom:0;'>MODEL PERFORMANCE ANALYSIS</h2>"
        "<p style='color:#94a3b8;letter-spacing:1px;'>Retrain on a custom period, "
        "evaluate on held-out matches, and export the results.</p></div>",
        unsafe_allow_html=True
    )

    # Feature importance charts (always from the saved ProductUI_Model)
    with st.container(border=True):
        st.markdown(
            "<h4 style='color:#00f0ff;margin-top:20px;margin-bottom:10px;'>"
            "🧠 KEY PREDICTIVE SIGNALS (XGBOOST FEATURE IMPORTANCE)</h4>",
            unsafe_allow_html=True
        )
        try:
            artifact    = get_model_artifact()
            fi_models   = artifact['models']
            fi_features = artifact['feature_cols']
            fi_col1, fi_col2 = st.columns(2)
            fi_col3, fi_col4 = st.columns(2)
            cols_list   = [fi_col1, fi_col2, fi_col3, fi_col4]
            role_colors = {'BAT': '#00f0ff', 'BOWL': '#f59e0b', 'AR': '#10b981', 'WK': '#8b5cf6'}
            for i, role in enumerate(['BAT', 'BOWL', 'AR', 'WK']):
                model  = fi_models[role]
                imp_df = pd.DataFrame({'Feature': fi_features,
                                       'Importance': model.feature_importances_})
                imp_df = imp_df.sort_values('Importance', ascending=True).tail(10)
                fig = px.bar(imp_df, x='Importance', y='Feature',
                             orientation='h', title=f"{role} MODEL")
                fig.update_traces(marker_color=role_colors[role], marker_line_width=0)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Rajdhani", size=14, color="#94a3b8"),
                    title=dict(font=dict(size=18, color=role_colors[role],
                                        family="Rajdhani")),
                    margin=dict(l=0, r=0, t=40, b=10), height=250,
                    xaxis=dict(showgrid=False, visible=False),
                    yaxis=dict(showgrid=False, title="")
                )
                with cols_list[i]:
                    with st.container(border=True):
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(
                f"Feature importance unavailable — ensure ProductUI_Model is trained. "
                f"Error: {e}"
            )

    # Date range inputs 
    with st.container(border=True):
        st.markdown(
            "<h4 style='color:#00f0ff;margin-top:10px;margin-bottom:10px;'>"
            "⏱️ DEFINE EVALUATION PERIODS</h4>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<hr style='border:1px solid #1e293b;margin:20px 0;'>",
            unsafe_allow_html=True
        )

        col_train1, col_train2 = st.columns(2)
        with col_train1:
            train_start = st.date_input(
                "TRAINING START DATE", value=datetime.date(2000, 1, 1)
            )
        with col_train2:
            train_end = st.date_input(
                "TRAINING END DATE",
                value=datetime.date(2024, 6, 30),
                max_value=datetime.date(2024, 6, 30),
                help="Cannot exceed 2024-06-30 (strict contest rule)"
            )

        st.markdown(
            "<hr style='border:1px solid #1e293b;margin:15px 0;'>",
            unsafe_allow_html=True
        )

        col_test1, col_test2 = st.columns(2)
        with col_test1:
            test_start = st.date_input(
                "TESTING START DATE", value=datetime.date(2024, 8, 1)
            )
        with col_test2:
            test_end = st.date_input(
                "TESTING END DATE", value=datetime.date(2024, 9, 22)
            )

    st.markdown("<br>", unsafe_allow_html=True)

    #The Retrain Toggle 
    force_retrain = st.toggle("🔄 Force Retrain Models (Check this if you changed the Training Dates)", value=False)

    # Run evaluation button
    if st.button("RUN XGBOOST EVALUATION", type="primary"):

        if test_start > test_end:
            st.error("Testing start date must be before or equal to testing end date.")
            st.stop()

        progress = st.progress(0, text="Loading dataset…")
        try:
            full_df = load_full_dataset("cleaned_dataset.csv")
        except FileNotFoundError:
            st.error("cleaned_dataset.csv not found. Place it in the same directory as app.py.")
            st.stop()

       
        # FAST EVALUATION (SAVED MODEL)
      
        if not force_retrain:
            progress.progress(50, text="Evaluating matches using saved ProductUI_Model…")
            results_df = generate_evaluation_csv(full_df, test_start, test_end)
            
            if results_df.empty:
                st.error("No valid matches found in the testing period (each match needs ≥11 players).")
                st.stop()

            model_display_name = "ProductUI_Model.pkl"
            model_display_sub = "Loaded from artifacts"
            
            progress.progress(100, text="Done!")

      
        # FULL RETRAINING (CUSTOM DATES)
       
        else:
            if train_end > datetime.date(2024, 6, 30):
                st.error("Training end date cannot exceed 2024-06-30. This is a strict contest rule.")
                st.stop()

            TARGET    = 'fantasy_points'
            DROP_COLS = ['match_id', 'date', 'venue', 'match_type', 'team', 'opposition', 'player', 'role', 'dismissal_kind']
            LEAKY_COLS = ['balls_per_boundary', 'boundary_percent', 'boundary_runs', 'bowling_sr', 'dot_ball_percent', 'dots', 'extras', 'fielding_points_raw', 'fifty_plus', 'five_wicket_haul', 'hundred', 'is_out', 'maiden_overs', 'overs', 'three_wicket_haul', 'venue_high_scoring']
            FEATURE_COLS_UI = [c for c in full_df.columns if c not in DROP_COLS + [TARGET] + LEAKY_COLS]
            ROLES_LIST = ['BAT', 'BOWL', 'AR', 'WK']

            BEST_PARAMS = {
                'BAT':  {'n_estimators': 408, 'max_depth': 8, 'learning_rate': 0.012925, 'subsample': 0.7776, 'colsample_bytree': 0.6359, 'min_child_weight': 6, 'random_state': 42, 'n_jobs': 1},
                'BOWL': {'n_estimators': 437, 'max_depth': 6, 'learning_rate': 0.018627, 'subsample': 0.7655, 'colsample_bytree': 0.7753, 'min_child_weight': 6, 'random_state': 42, 'n_jobs': 1},
                'AR':   {'n_estimators': 378, 'max_depth': 7, 'learning_rate': 0.011623, 'subsample': 0.8169, 'colsample_bytree': 0.6415, 'min_child_weight': 2, 'random_state': 42, 'n_jobs': 1},
                'WK':   {'n_estimators': 674, 'max_depth': 6, 'learning_rate': 0.014482, 'subsample': 0.8677, 'colsample_bytree': 0.9286, 'min_child_weight': 9, 'random_state': 42, 'n_jobs': 1},
            }

            retrain_df = full_df[(full_df['date'] >= pd.Timestamp(train_start)) & (full_df['date'] <= pd.Timestamp(train_end))].copy()
            eval_full_df = full_df[(full_df['date'] >= pd.Timestamp(test_start)) & (full_df['date'] <= pd.Timestamp(test_end))].copy()

            if retrain_df.empty:
                st.error("No training data found for the specified training period.")
                st.stop()

            progress.progress(10, text=f"Training on {len(retrain_df):,} rows…")

            newly_trained = {}
            for k, role in enumerate(ROLES_LIST):
                role_data = retrain_df[retrain_df['role'] == role]
                if not role_data.empty:
                    xgb_model = XGBRegressor(**BEST_PARAMS[role])
                    xgb_model.fit(role_data[FEATURE_COLS_UI].fillna(0), role_data[TARGET])
                    newly_trained[role] = xgb_model
                progress.progress(10 + (k + 1) * 15, text=f"Trained {role} model…")

            os.makedirs('src/model_artifacts', exist_ok=True)
            model_filename = f"src/model_artifacts/model_{train_end.strftime('%Y-%m-%d')}.pkl"
            
            with open(model_filename, 'wb') as mf:
                pickle.dump({'models': newly_trained, 'feature_cols': FEATURE_COLS_UI, 'training_cutoff': str(train_end)}, mf)

            progress.progress(75, text="Evaluating on test matches…")

            results = []
            for match_id in eval_full_df['match_id'].unique():
                match_data = eval_full_df[eval_full_df['match_id'] == match_id].copy().reset_index(drop=True)
                if len(match_data) < 11: continue

                squad = match_data.copy()
                if 'WK' not in squad['role'].values:
                    bat_rows = squad[squad['role'] == 'BAT']
                    if not bat_rows.empty: squad.loc[bat_rows.index[0], 'role'] = 'WK'

                squad['predicted_fp'] = 0.0
                for role in squad['role'].unique():
                    role_mask = squad['role'] == role
                    mdl = newly_trained.get(role, newly_trained.get('BAT'))
                    if mdl:
                        preds = mdl.predict(squad.loc[role_mask, FEATURE_COLS_UI].fillna(0))
                        squad.loc[role_mask, 'predicted_fp'] = np.maximum(0.0, np.round(preds, 2))

                pred_team_eval = select_team_ilp(squad[['player', 'team', 'role', 'predicted_fp']], problem_name="eval")
                actual_team = get_actual_dream_team(match_data)

                fp_map = match_data.set_index('player')['fantasy_points'].to_dict()
                pred_total = sum(fp_map.get(p, 0) for p in pred_team_eval['player'])
                dream_total = actual_team['predicted_fp'].sum()

                row = {
                    'Match Date': match_data['date'].iloc[0].strftime('%Y-%m-%d'),
                    'Team 1': match_data['team'].iloc[0],
                    'Team 2': match_data['opposition'].iloc[0],
                }

                pred_sorted = pred_team_eval.sort_values('predicted_fp', ascending=False)
                actual_sorted = actual_team.sort_values('predicted_fp', ascending=False)

                for i in range(11):
                    row[f'Predicted Player {i+1}'] = pred_sorted['player'].iloc[i] if i < len(pred_sorted) else "N/A"
                    row[f'Predicted Player {i+1} Points'] = float(pred_sorted['predicted_fp'].iloc[i]) if i < len(pred_sorted) else 0.0
                    row[f'Dream Team Player {i+1}'] = actual_sorted['player'].iloc[i] if i < len(actual_sorted) else "N/A"
                    row[f'Dream Team Player {i+1} Points'] = float(actual_sorted['predicted_fp'].iloc[i]) if i < len(actual_sorted) else 0.0

                row['Total Predicted Points'] = round(pred_total, 2)
                row['Total Dream Team Points'] = round(dream_total, 2)
                row['MAE'] = round(abs(dream_total - pred_total), 2)
                results.append(row)

            results_df = pd.DataFrame(results)
            model_display_name = f"model_{train_end.strftime('%Y-%m-%d')}.pkl"
            model_display_sub = f"Retrained on {train_start} → {train_end}"
            progress.progress(100, text="Done!")


        
        # RENDER UI (Runs for both paths)
      
        overall_mae = results_df['MAE'].mean()
        n_matches   = len(results_df)

        st.markdown(
            f"<div style='display:flex;gap:20px;margin-top:20px;margin-bottom:30px;'>"
            f"<div class='glass-box' style='flex:1;text-align:center;border-bottom:4px solid #00f0ff;'>"
            f"<span style='color:#94a3b8;font-size:14px;font-weight:bold;letter-spacing:2px;'>MEAN MAE (POINTS GAP)</span><br>"
            f"<span style='font-size:36px;color:#00f0ff;font-weight:bold;'>{overall_mae:.2f}</span><br>"
            f"<span style='color:#64748b;font-size:12px;'>Dream Team pts − Predicted Team pts</span></div>"
            f"<div class='glass-box' style='flex:1;text-align:center;border-bottom:4px solid #f59e0b;'>"
            f"<span style='color:#94a3b8;font-size:14px;font-weight:bold;letter-spacing:2px;'>VALIDATED FIXTURES</span><br>"
            f"<span style='font-size:36px;color:#f59e0b;font-weight:bold;'>{n_matches}</span><br>"
            f"<span style='color:#64748b;font-size:12px;'>Strict Out-of-Sample Testing</span></div>"
            f"<div class='glass-box' style='flex:1;text-align:center;border-bottom:4px solid #10b981;'>"
            f"<span style='color:#94a3b8;font-size:14px;font-weight:bold;letter-spacing:2px;'>ACTIVE MODEL</span><br>"
            f"<span style='font-size:15px;color:#10b981;font-weight:bold;word-break:break-all;'>{model_display_name}</span><br>"
            f"<span style='color:#64748b;font-size:12px;'>{model_display_sub}</span></div>"
            f"</div>",
            unsafe_allow_html=True
        )

        if force_retrain:
            st.success(f"Custom Training Complete! New weights evaluated successfully.")
        else:
            st.success("Fast Evaluation Complete! Used standard competition model weights.")

      

        # Scatter plot
        st.markdown(
            "<h4 style='color:#00f0ff;margin-top:30px;margin-bottom:10px;'>"
            "📊 PREDICTION DEVIATION ANALYSIS</h4>",
            unsafe_allow_html=True
        )
        with st.container(border=True):
            fig_scatter = px.scatter(
                results_df,
                x='Total Predicted Points',
                y='Total Dream Team Points',
                hover_data=['Match Date', 'Team 1', 'Team 2', 'MAE'],
                labels={
                    'Total Predicted Points':  'XGBoost Predicted Squad Points',
                    'Total Dream Team Points': 'Actual Optimal Squad Points'
                }
            )
            min_val = min(
                results_df['Total Predicted Points'].min(),
                results_df['Total Dream Team Points'].min()
            ) * 0.95
            max_val = max(
                results_df['Total Predicted Points'].max(),
                results_df['Total Dream Team Points'].max()
            ) * 1.05
            fig_scatter.add_shape(
                type="line", line=dict(dash="dash", color="#f59e0b", width=2),
                x0=min_val, y0=min_val, x1=max_val, y1=max_val
            )
            fig_scatter.update_traces(
                marker=dict(size=12, color="#00f0ff",
                            line=dict(width=2, color="#1e293b")),
                opacity=0.8
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(16,24,39,0.4)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Rajdhani", size=14, color="#94a3b8"),
                xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
                yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
                margin=dict(l=10, r=10, t=20, b=10), height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown(
                "<p style='text-align:center;color:#64748b;font-size:13px;"
                "margin-top:-15px;'>*Dots closer to the amber dashed line "
                "indicate higher prediction accuracy.</p>",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # CSV download
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 EXPORT EVALUATION MATRIX (CSV)",
            data=csv_data,
            file_name=f"model_evaluation_{test_start}_{test_end}.csv",
            mime='text/csv',
            type="primary"
        )
