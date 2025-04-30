import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import poisson

# -------------------------------
# CONFIGURATION
# -------------------------------
RAPIDAPI_KEY = "4b4bbc76bamsha135115d4bdeea0p13b9bajsn1944f47c1f96"  # Replace with your key
HEADERS = {
    "x-rapidapi-host": "api-football-v1.p.rapidapi.com",
    "x-rapidapi-key": RAPIDAPI_KEY
}
BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"
LEAGUE_ID = 39  # Premier League
SEASON = 2023

# -------------------------------
# HELPERS
# -------------------------------
def fetch_team_stats(team_id):
    url = f"{BASE_URL}/teams/statistics"
    params = {"league": LEAGUE_ID, "season": SEASON, "team": team_id}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        return None
    return response.json().get("response")

def calculate_expected_goals(home_stats, away_stats):
    try:
        home_attack = home_stats['goals']['for']['average']['home']
        away_defense = away_stats['goals']['against']['average']['away']
        away_attack = away_stats['goals']['for']['average']['away']
        home_defense = home_stats['goals']['against']['average']['home']

        home_xg = float(home_attack) * float(away_defense)
        away_xg = float(away_attack) * float(home_defense)

        return home_xg, away_xg
    except (KeyError, TypeError, ValueError):
        return 1.0, 1.0  # fallback default

def poisson_matrix(home_goals, away_goals, max_goals=5):
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i, j] = poisson.pmf(i, home_goals) * poisson.pmf(j, away_goals)
    return matrix

def fetch_teams():
    url = f"{BASE_URL}/teams"
    params = {"league": LEAGUE_ID, "season": SEASON}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        return []
    teams = response.json().get("response", [])
    return [(team["team"]["id"], team["team"]["name"]) for team in teams]

# -------------------------------
# STREAMLIT APP
# -------------------------------
st.title("⚽ Football Match Predictor")

teams = fetch_teams()
team_dict = dict(teams)
team_names = [name for _, name in teams]

col1, col2 = st.columns(2)
home_team_name = col1.selectbox("Select Home Team", team_names)
away_team_name = col2.selectbox("Select Away Team", team_names, index=1)

if home_team_name == away_team_name:
    st.warning("Home and away teams must be different.")
else:
    home_team_id = [tid for tid, name in teams if name == home_team_name][0]
    away_team_id = [tid for tid, name in teams if name == away_team_name][0]

    st.write(f"Getting stats for **{home_team_name}** vs **{away_team_name}**...")
    home_stats = fetch_team_stats(home_team_id)
    away_stats = fetch_team_stats(away_team_id)

    if not home_stats or not away_stats:
        st.error("Could not fetch stats. Check your API key or try again later.")
    else:
        home_xg, away_xg = calculate_expected_goals(home_stats, away_stats)

        st.subheader("Expected Goals")
        st.write(f"{home_team_name}: {home_xg:.2f}  |  {away_team_name}: {away_xg:.2f}")

        matrix = poisson_matrix(home_xg, away_xg)
        df_matrix = pd.DataFrame(matrix, columns=[f"{away_team_name} {i}" for i in range(6)],
                                 index=[f"{home_team_name} {i}" for i in range(6)])

        st.subheader("Probability Matrix")
        st.dataframe(df_matrix.style.background_gradient(cmap='Blues'))

        most_prob_index = np.unravel_index(np.argmax(matrix), matrix.shape)
        st.success(f"Most probable scoreline: {home_team_name} {most_prob_index[0]} - {away_team_name} {most_prob_index[1]}")

        # Optionally, save to CSV
        if st.button("Export CSV"):
            csv_path = os.path.join(os.path.expanduser("~"), "Desktop", f"{home_team_name}_vs_{away_team_name}_prediction.csv")
            df_matrix.to_csv(csv_path)
            st.success(f"Prediction matrix saved to {csv_path}")

