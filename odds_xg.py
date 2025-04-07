import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import math
import numpy as np
import random
import time
from scipy.stats import norm, poisson, skellam
from scipy.optimize import minimize

#Set page title and icon
st.set_page_config(page_title="BetWise", page_icon=":soccer:")

# Calculate 1x2 and xG
def calculate_1x2_and_xg(home_xg, away_xg, max_goals=10):
    if home_xg < 0 or away_xg < 0:
        raise ValueError("Invalid inputs: xG values must be non-negative")

    # Calculate probabilities using Poisson distribution
    p_home_win = 0.0
    p_draw = 0.0
    p_away_win = 0.0

    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            p = poisson.pmf(home_goals,home_xg) * poisson.pmf(away_goals, away_xg)
            if home_goals > away_goals:
                p_home_win += p
            elif home_goals == away_goals:
                p_draw += p
            else:
                p_away_win += p

    # Normalize to ensure probabilities sum to 1 (due to truncation)
    total = p_home_win + p_draw + p_away_win
    if total > 0:
        p_home_win /= total
        p_draw /= total
        p_away_win /= total

    return p_home_win, p_draw, p_away_win

def calculate_xg_from_dnb_probs(home_dnb_prob, away_dnb_prob, total_xg, max_iter=500):
    """
    Calculate home/away xG directly from DNB probabilities and total xG.

    Args:
        home_dnb_prob (float): Probability of home win (draws excluded) [0-1]
        away_dnb_prob (float): Probability of away win (draws excluded) [0-1]
        total_xg (float): Total expected goals in the match
        max_iter (int): Maximum optimization iterations

    Returns:
        tuple: (home_xg, away_xg)
    """
    # Validate probabilities
    if not np.isclose(home_dnb_prob + away_dnb_prob, 1.0, atol=1e-4):
        raise ValueError("DNB probabilities must sum to 1")

    def objective(lambda_h):
        lambda_a = total_xg - lambda_h

        if lambda_h <= 1e-5 or lambda_a <= 1e-5:
            return 1e9  # Penalize invalid values

        # Calculate outcome probabilities using Skellam distribution
        home_win_prob = 1 - skellam.cdf(0, lambda_h, lambda_a)
        draw_prob = skellam.pmf(0, lambda_h, lambda_a)

        # Avoid division by zero in extreme cases
        if draw_prob >= 1 - 1e-5:
            return 1e9

        # Calculate model's DNB probabilities
        model_h_dnb = home_win_prob / (1 - draw_prob)
        model_a_dnb = (1 - draw_prob - home_win_prob) / (1 - draw_prob) # Corrected away DNB prob

        # Calculate error (squared differences)
        error = (model_h_dnb - home_dnb_prob)**2 + (model_a_dnb - away_dnb_prob)**2
        return error

    # Initial guess based on DNB probabilities
    initial_guess = total_xg * home_dnb_prob
    bounds = [(1e-5, total_xg - 1e-5)]

    # Numerical optimization
    result = minimize(objective,
                     x0=initial_guess,
                     method='L-BFGS-B',
                     bounds=bounds,
                     options={'maxiter': max_iter})

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    home_xg = round(result.x[0], 4)
    away_xg = round(total_xg - home_xg, 4)
    return home_xg, away_xg


#Dictionaries of country and leagues
leagues_dict = {
    "England": ["UK1", "UK2", "UK3", "UK4", "UK5", "UK6N", "UK6S", "UK7N"],
    "Germany": ["DE1", "DE2", "DE3", "DE4SW", "DE4W", "DE4N", "DE4NO", "DE4B"],
    "Italy": ["IT1", "IT2", "IT3C", "IT3B", "IT3A"],
    "Spain": ["ES1", "ES2", "ES3G1", "ES3G2", "ES3G3", "ES3G4", "ES3G5"],
    "France": ["FR1", "FR2", "FR3"],
    "Sweden": ["SW1", "SW2", "SW3S", "SW3N"],
    "Netherlands": ["NL1", "NL2", "NL3"],
    "Russia": ["RU1", "RU2"],
    "Portugal": ["PT1", "PT2"],
    "Austria": ["AT1", "AT2", "AT3O", "AT3T", "AT3M", "AT3W", "AT3V"],
    "Denmark": ["DK1", "DK2", "DK3G1", "DK3G2"],
    "Greece": ["GR1", "GR2"],
    "Norway": ["NO1", "NO2", "NO3G1", "NO3G2"],
    "Czech-Republic": ["CZ1", "CZ2"],
    "Turkey": ["TU1", "TU2", "TU3B", "TU3K"],
    "Belgium": ["BE1", "BE2"],
    "Scotland": ["SC1", "SC2", "SC3", "SC4"],
    "Switzerland": ["CH1", "CH2"],
    "Finland": ["FI1", "FI2", "FI3A", "FI3B", "FI3C"],
    "Ukraine": ["UA1", "UA2"],
    "Romania": ["RO1", "RO2"],
    "Poland": ["PL1", "PL2", "PL3"],
    "Croatia": ["HR1", "HR2"],
    "Belarus": ["BY1", "BY2"],
    "Israel": ["IL1", "IL2"],
    "Iceland": ["IS1", "IS2", "IS3", "IS4"],
    "Cyprus": ["CY1", "CY2"],
    "Serbia": ["CS1", "CS2"],
    "Bulgaria": ["BG1", "BG2"],
    "Slovakia": ["SK1", "SK2"],
    "Hungary": ["HU1", "HU2"],
    "Kazakhstan": ["KZ1", "KZ2"],
    "Bosnia-Herzegovina": ["BA1"],
    "Slovenia": ["SI1", "SI2"],
    "Azerbaijan": ["AZ1"],
    "Ireland": ["IR1", "IR2"],
    "Latvia": ["LA1", "LA2"],
    "Georgia": ["GE1", "GE2"],
    "Kosovo": ["XK1"],
    "Albania": ["AL1"],
    "Lithuania": ["LT1", "LT2"],
    "North-Macedonia": ["MK1"],
    "Armenia": ["AM1"],
    "Estonia": ["EE1", "EE2"],
    "Northern-Ireland": ["NI1", "NI2"],
    "Malta": ["MT1"],
    "Luxembourg": ["LU1"],
    "Wales": ["WL1"],
    "Montenegro": ["MN1"],
    "Moldova": ["MD1"],
    "Färöer": ["FA1"],
    "Gibraltar": ["GI1"],
    "Andorra": ["AD1"],
    "San-Marino": ["SM1"],
    "Brazil": ["BR1", "BR2", "BR3", "BRC", "BRGA"],
    "Mexico": ["MX1", "MX2"],
    "Argentina": ["AR1", "AR2", "AR3F", "AR5", "AR3", "AR4"],
    "USA": ["US1", "US2", "US3"],
    "Colombia": ["CO1", "CO2"],
    "Ecuador": ["EC1", "EC2"],
    "Paraguay": ["PY1", "PY2"],
    "Chile": ["CL1", "CL2"],
    "Uruguay": ["UY1", "UY2"],
    "Costa-Rica": ["CR1", "CR2"],
    "Bolivia": ["BO1"],
    "Guatemala": ["GT1", "GT2"],
    "Dominican-Rep.": ["DO1"],
    "Honduras": ["HN1"],
    "Venezuela": ["VE1"],
    "Peru": ["PE1", "PE2"],
    "Panama": ["PA1"],
    "El-Salvador": ["SV1"],
    "Jamaica": ["JM1"],
    "Nicaragua": ["NC1"],
    "Canada": ["CA1"],
    "Haiti": ["HT1"],
    "Japan": ["JP1", "JP2", "JP3"],
    "South-Korea": ["KR1", "KR2", "KR3"],
    "China": ["CN1", "CN2", "CN3"],
    "Iran": ["IA1", "IA2"],
    "Australia": ["AU1", "AU2V", "AU2NSW", "AU2Q", "AU2S", "AU2W"],
    "Saudi-Arabia": ["SA1", "SA2"],
    "Thailand": ["TH1", "TH2"],
    "Qatar": ["QA1", "QA2"],
    "United-Arab-Emirates": ["AE1", "AE2"],
    "Indonesia": ["ID1", "ID2"],
    "Jordan": ["JO1"],
    "Syria": ["SY1"],
    "Uzbekistan": ["UZ1"],
    "Malaysia": ["MY1", "MY2"],
    "Vietnam": ["VN1", "VN2"],
    "Iraq": ["IQ1"],
    "Kuwait": ["KW1"],
    "Bahrain": ["BH1"],
    "Myanmar": ["MM1"],
    "Palestine": ["PS1"],
    "India": ["IN1", "IN2"],
    "New-Zealand": ["NZ1"],
    "Hong-Kong": ["HK1", "HK2"],
    "Oman": ["OM1"],
    "Taiwan": ["TW1"],
    "Tajikistan": ["TJ1"],
    "Turkmenistan": ["TM1"],
    "Lebanon": ["LB1"],
    "Bangladesh": ["BD1"],
    "Singapore": ["SG1"],
    "Cambodia": ["KH1"],
    "Kyrgyzstan": ["KG1"],
    "Egypt": ["EG1", "EG2"],
    "Algeria": ["DZ1", "DZ2"],
    "Tunisia": ["TN1", "TN2"],
    "Morocco": ["MA1", "MA2"],
    "South-Africa": ["ZA1", "ZA2"],
    "Kenya": ["KE1", "KE2"],
    "Zambia": ["ZM1"],
    "Ghana": ["GH1"],
    "Nigeria": ["NG1"],
    "Uganda": ["UG1"],
    "Burundi": ["BI1"],
    "Rwanda": ["RW1"],
    "Cameroon": ["CM1"],
    "Tanzania": ["TZ1"],
    "Gambia": ["GM1"],
    "Sudan": ["SD1"]
}

# List of spinner messages
spinner_messages = [
    "Fetching the latest football ratings...",
    "Hold tight, we're gathering the data...",
    "Just a moment, crunching the numbers...",
    "Loading the football magic...",
    "Almost there, preparing the stats..."
]

# Function to fetch table from website
def fetch_table(country, league, table_type="home"):
    url = f"https://www.soccer-rating.com/{country}/{league}/{table_type}/"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        html_io = io.StringIO(str(soup))
        tables = pd.read_html(html_io, flavor="lxml")

        # Get the rating table as before (using table index 14)
        rating_table = tables[14] if tables and len(tables) > 14 else None

        # Expected columns for the league table
        expected_columns = {"Home", "Away", "Home.4", "Away.4"}
        possible_indices = [28, 24, 23]
        league_table = None
        for idx in possible_indices:
            if tables and len(tables) > idx:
                candidate = tables[idx]
                candidate_cols = set(candidate.columns.astype(str))
                if expected_columns.issubset(candidate_cols):
                    league_table = candidate
                    break
        # Fallback: search all tables for expected columns
        if league_table is None:
            for candidate in tables:
                candidate_cols = set(candidate.columns.astype(str))
                if expected_columns.issubset(candidate_cols):
                    league_table = candidate
                    break
        return rating_table, league_table
    except Exception as e:
        return None, None

# Custom CSS for styling
st.markdown("""\
    <style>
        body {
            background-color: #f4f4f9;
            font-family: 'Arial', sans-serif;
        }
        .header {
            font-size: 32px;
            color: #3b5998;
            font-weight: bold;
            text-align: center;
        }
        .section-header {
            font-size: 20px;
            font-weight: 600;
            color: #007BFF;
        }
        .subsection-header {
            font-size: 18px;
            font-weight: 500;
            color: #5a5a5a;
        }
        .rating-table th {
            background-color: #007BFF;
            color: white;
            text-align: center;
        }
        .rating-table td {
            text-align: center;
        }
        .win-probability {
            color: #28a745;
            font-size: 18px;
            font-weight: 600;
        }
        .odds {
            color: #dc3545;
            font-size: 18px;
            font-weight: 600;
        }
        .slider {
            margin-top: 20px;
            padding: 10px;
            border-radius: 10px;
            background-color: #007BFF;
            color: white;
        }
        .button {
            background-color: #28a745;
            color: white;
            padding: 10px;
            border-radius: 8px;
            font-size: 16px;
        }
        .button:hover {
            background-color: #218838;
        }
        .card {
            background-color: #f8f9fa;
            border: 1px solid #007BFF;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin: 10px;
            transition: transform 0.2s;
        }
        .card:hover {
            transform: scale(1.05);
        }
        .card-title {
            color: #007BFF;
            font-weight: bold;
            font-size: 18px;
        }
        .card-odds {
            font-size: 24px;
            font-weight: bold;
            color: red;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit header
st.markdown('<div class="header">⚽ BetWise- Elo Odds Calculator</div>', unsafe_allow_html=True)

# Explanation tooltip
if "data_fetched" not in st.session_state:
    st.info("Use the sidebar to select a country and league. Click 'Get Ratings' to fetch the latest data.")

# Sidebar: How to use
with st.sidebar.expander("How to Use This App", expanded=True):
    st.write("1. Select Country and League.")
    st.write("2. Click 'Get Ratings' to fetch the latest data.")
    st.write("3. Select Home and Away Teams from the dropdowns.")
    st.write("4. View calculated odds and expected goals.")


# Sidebar: Select Match Details
st.sidebar.header("⚽ Select Match Details")
selected_country = st.sidebar.selectbox("Select Country:", list(leagues_dict.keys()), index=0)
selected_league = st.sidebar.selectbox("Select League:", leagues_dict[selected_country], index=0)

# Create two tabs
tab1, tab2 = st.tabs(["Elo Ratings Odds Calculator", "League Table"])

with tab1:
    # Create a progress bar
    progress_bar = st.progress(0)

    # Fetch data if not available or league has changed
    if "home_table" not in st.session_state or "away_table" not in st.session_state or st.session_state.get("selected_league") != selected_league:
        if st.sidebar.button("Get Ratings", key="fetch_button", help="Fetch ratings and tables for selected country and league"):
            with st.spinner(random.choice(spinner_messages)):
                for i in range(100):  # Simulate progress
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)
                home_table, home_league_table = fetch_table(selected_country, selected_league, "home")
                away_table, away_league_table = fetch_table(selected_country, selected_league, "away")
                progress_bar.empty()
                if isinstance(home_table, pd.DataFrame) and isinstance(away_table, pd.DataFrame):
                    home_table = home_table.drop(home_table.columns[[0, 2, 3]], axis=1)
                    away_table = away_table.drop(away_table.columns[[0, 2, 3]], axis=1)
                    st.session_state["home_table"] = home_table
                    st.session_state["away_table"] = away_table
                    st.session_state["league_table"] = home_league_table  # Store the league table
                    st.session_state["selected_league"] = selected_league
                    st.success("Data fetched successfully!")
                else:
                    st.error("Error fetching one or both tables. Please try again.")

    # Display team selection and ratings if data is available
    if "home_table" in st.session_state and "away_table" in st.session_state:
        st.markdown('<div class="section-header">⚽ Match Details</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Select Home Team:", st.session_state["home_table"].iloc[:, 0])
        with col2:
            away_team = st.selectbox("Select Away Team:", st.session_state["away_table"].iloc[:, 0])

        # Fetch team ratings
        home_team_data = st.session_state["home_table"][st.session_state["home_table"].iloc[:, 0] == home_team]
        away_team_data = st.session_state["away_table"][st.session_state["away_table"].iloc[:, 0] == away_team]
        home_rating = home_team_data.iloc[0, 1]
        away_rating = away_team_data.iloc[0, 1]
        home = 10**(home_rating / 400)
        away = 10**(away_rating / 400)
        home_win_prob_raw = home / (home + away)
        away_win_prob_raw = away / (home + away)

        # Normalize win probabilities to exclude draw for DNB calculation
        total_win_prob = home_win_prob_raw + away_win_prob_raw
        home_win_prob_dnb = home_win_prob_raw / total_win_prob if total_win_prob > 0 else 0.5 # Normalize for DNB
        away_win_prob_dnb = away_win_prob_raw / total_win_prob if total_win_prob > 0 else 0.5 # Normalize for DNB


        # Display Ratings and Win Probabilities
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"{home_team} Home Rating: {home_rating}")
        with col2:
            st.write(f"{away_team} Away Rating: {away_rating}")
        st.markdown('<div class="section-header">Win Probability</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{home_team} Win Probability:** {home_win_prob_raw:.2f}")
        with col2:
            st.write(f"**{away_team} Win Probability:** {away_win_prob_raw:.2f}")

        # Draw No Bet Odds Calculation
        home_draw_no_bet_odds = 1 / home_win_prob_dnb if home_win_prob_dnb > 0 else float('inf')
        away_draw_no_bet_odds = 1 / away_win_prob_dnb if away_win_prob_dnb > 0 else float('inf')
        st.markdown('<div class="section-header">Draw No Bet Odds</div>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)
        with col3:
            st.write(f"**{home_team} Draw No Bet Odds:** {home_draw_no_bet_odds:.2f}")
        with col4:
            st.write(f"**{away_team} Draw No Bet Odds:** {away_draw_no_bet_odds:.2f}")

        # Initialize variables for goals statistics
        home_goals_for_per_game = None
        home_goals_against_per_game = None
        away_goals_for_per_game = None
        away_goals_against_per_game = None

        # Helper function to extract goals (format "GF:GA")
        def extract_goals_parts(value):
            try:
                parts = value.split(":")
                if len(parts) >= 2:
                    goals_for = float(parts[0].strip())
                    goals_against = float(parts[1].strip())
                    return goals_for, goals_against
                else:
                    return None, None
            except Exception as e:
                return None, None

        # Calculate goals statistics from league table if available
        if "league_table" in st.session_state and st.session_state["league_table"] is not None:
            league_table = st.session_state["league_table"]
            # Home team stats
            home_team_row = league_table[league_table.iloc[:, 1] == home_team]
            if not home_team_row.empty:
                home_raw = home_team_row.iloc[0]["Home.4"]
                home_goals_for, home_goals_against = extract_goals_parts(home_raw)
                try:
                    home_games = float(home_team_row.iloc[0]["Home"])
                    if home_games and home_games != 0:
                        if home_goals_for is not None:
                            home_goals_for_per_game = home_goals_for / home_games
                        if home_goals_against is not None:
                            home_goals_against_per_game = home_goals_against / home_games
                except Exception as e:
                    pass
            # Away team stats
            away_team_row = league_table[league_table.iloc[:, 1] == away_team]
            if not away_team_row.empty:
                away_raw = away_team_row.iloc[0]["Away.4"]
                away_goals_for, away_goals_against = extract_goals_parts(away_raw)
                try:
                    away_games = float(away_team_row.iloc[0]["Away"])
                    if away_games and away_games != 0:
                        if away_goals_for is not None:
                            away_goals_for_per_game = away_goals_for / away_games
                        if away_goals_against is not None:
                            away_goals_against_per_game = away_goals_against / away_games
                except Exception as e:
                    pass
        # Calculate average league goals per match using 'Goals' (format "GF:GA") and 'M' (matches)
        avg_goals_per_match = None
        if "league_table" in st.session_state and st.session_state["league_table"] is not None:
            league_table = st.session_state["league_table"]
            if "Goals" in league_table.columns and "M" in league_table.columns:
                league_table["GF"] = league_table["Goals"].apply(lambda x: float(x.split(":")[0].strip()) if isinstance(x, str) and ":" in x else None)
                league_table["GA"] = league_table["Goals"].apply(lambda x: float(x.split(":")[1].strip()) if isinstance(x, str) and ":" in x else None)
                avg_GF = league_table["GF"].mean()
                avg_GA = league_table["GA"].mean()
                avg_total = (league_table["GF"] + league_table["GA"]).mean()
                avg_matches = league_table["M"].mean()
                if avg_matches and avg_matches != 0:
                    avg_goals_per_match = avg_total / avg_matches
                st.markdown('<div class="section-header">League Average Goals</div>', unsafe_allow_html=True)
                if avg_goals_per_match is not None:
                    st.write(f"**Average Goals per Match:** {avg_goals_per_match:.2f}")
                else:
                    st.write("**Average Goals per Match:** N/A")
            else:
                st.write("The required columns ('Goals' and/or 'M') were not found in the league table.")

        # Calculate Expected Goals using per game statistics
        home_xg_base = None
        away_xg_base = None
        total_expected_goals = None
        if home_goals_for_per_game is not None and away_goals_against_per_game is not None:
            home_xg_base = (home_goals_for_per_game + away_goals_against_per_game) / 2
        if away_goals_for_per_game is not None and home_goals_against_per_game is not None:
            away_xg_base = (away_goals_for_per_game + home_goals_against_per_game) / 2
        if home_xg_base is not None and away_xg_base is not None and avg_goals_per_match is not None:
            total_expected_goals = ((home_xg_base + away_xg_base) + avg_goals_per_match) / 2

        # Calculate xG from DNB probs and total xG
        home_xg = None
        away_xg = None
        if total_expected_goals is not None and total_expected_goals > 0:
            try:
                home_xg, away_xg = calculate_xg_from_dnb_probs(home_win_prob_dnb, away_win_prob_dnb, total_expected_goals)
            except ValueError as e:
                st.error(f"Error calculating xG from DNB probabilities: {e}")


        # Display 1X2 odds and xG
        if home_xg is not None and away_xg is not None and total_expected_goals is not None and total_expected_goals > 0:
            try:
                p_home_win, p_draw, p_away_win = calculate_1x2_and_xg(home_xg, away_xg)
                home_odds_poisson = 1 / p_home_win if p_home_win > 0 else float('inf')
                draw_odds_poisson = 1 / p_draw if p_draw > 0 else float('inf')
                away_odds_poisson = 1 / p_away_win if p_away_win > 0 else float('inf')
               
                p_under_25 = poisson.cdf(2, total_expected_goals)
                p_over_25 = 1 - p_under_25
                over_odds_poisson = 1 / p_over_25 if p_over_25 > 0 else float('inf')
                under_odds_poisson = 1 / p_under_25 if p_under_25 > 0 else float('inf')

            except ValueError as e:
                st.error(f"Error calculating 1X2 odds: {e}")
        st.markdown('<div class="section-header">1X2 Betting Odds</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div style='color:black' class='card'><b>{home_team}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><div class='card-title'></div><div class='card-odds'>{home_odds_poisson:.2f}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='color:black' class='card'><b>Draw</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><div class='card-title'></div><div class='card-odds'>{draw_odds_poisson:.2f}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div style='color:black' class='card'><b>{away_team}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><div class='card-title'></div><div class='card-odds'>{away_odds_poisson:.2f}</div></div>", unsafe_allow_html=True)

        #Display home xG, away xG and total xG
        st.markdown('<div class="section-header">Expected Goals</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='card'><div class='card-title'>{home_team} xG</div><div class='card-odds'>{home_xg:.2f}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='card'><div class='card-title'>{away_team} xG</div><div class='card-odds'>{away_xg:.2f}</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='card'><div class='card-title'>Total xG</div><div class='card-odds'>{total_expected_goals:.2f}</div></div>", unsafe_allow_html=True)

        #Display O/U 2.5 Odds
        st.markdown('<div class="section-header">Over/Under 2.5 Goals</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='card'><div class='card-title'>Over 2.5</div><div class='card-odds'>{over_odds_poisson:.2f}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='card'><div class='card-title'>Under 2.5</div><div class='card-odds'>{under_odds_poisson:.2f}</div></div>", unsafe_allow_html=True)

with tab2:
    # Display the league table as a simple text list
    if "league_table" in st.session_state and st.session_state["league_table"] is not None:
        league_table = st.session_state["league_table"]
        league_table.rename(columns={'Unnamed: 0': 'Position'}, inplace=True)
        for index, row in league_table.iterrows():
            team_name = row[league_table.columns[1]]
            points = row["P."]  # Points from the last column
            if pd.notna(team_name):
                st.write(f"{row['Position']:.0f}. {team_name} - Points: {points}")
