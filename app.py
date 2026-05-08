import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Advanced Game Recommendation System",
    page_icon="🎮",
    layout="wide"
)

# ---------------- LOAD DATA ---------------- #

games = pd.read_csv("advanced_hybrid_games_dataset.csv")

games = games.fillna('')

# ---------------- FEATURE ENGINEERING ---------------- #

games['features'] = (
    games['Genre'] + ' ' +
    games['Platform'] + ' ' +
    games['Mode'] + ' ' +
    games['Style'] + ' ' +
    games['Difficulty'] + ' ' +
    games['Developer'] + ' ' +
    games['Graphics Type'] + ' ' +
    games['Game Length'] + ' ' +
    games['Multiplayer Type']
)

# ---------------- TEXT VECTORIZATION ---------------- #

cv = CountVectorizer()

feature_matrix = cv.fit_transform(games['features'])

# ---------------- COSINE SIMILARITY ---------------- #

similarity = cosine_similarity(feature_matrix)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #0B1120, #111827);
    color: white;
}

/* Main Title */

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: bold;
    color: #60A5FA;
    margin-top: 10px;
}

/* Subtitle */

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #CBD5E1;
    margin-bottom: 30px;
}

/* Info Cards */

.info-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Button */

.stButton>button {
    background: linear-gradient(to right, #2563EB, #3B82F6);
    color: white;
    border-radius: 10px;
    padding: 12px 24px;
    border: none;
    font-size: 16px;
    font-weight: 600;
    width: 100%;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
    '<div class="main-title">🎮 Advanced Game Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Hybrid Content-Based Recommendation Using Cosine Similarity & Weighted Ranking</div>',
    unsafe_allow_html=True
)

# ---------------- INFO SECTION ---------------- #

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="info-card">
        <h3>Total Games</h3>
        <h2>{len(games)}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>Recommendation Type</h3>
        <h2>Hybrid Content-Based</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>Algorithm</h3>
        <h2>Cosine Similarity</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("🎯 User Preferences")

selected_game = st.sidebar.selectbox(
    "Select Favorite Game",
    games['Game Name'].values
)

preferred_platform = st.sidebar.selectbox(
    "Preferred Platform",
    sorted(games['Platform'].unique())
)

preferred_mode = st.sidebar.selectbox(
    "Preferred Mode",
    sorted(games['Mode'].unique())
)

minimum_rating = st.sidebar.slider(
    "Minimum Rating",
    0.0,
    10.0,
    7.0
)

# ---------------- RECOMMENDATION FUNCTION ---------------- #

def recommend_games(game_name):

    game_index = games[games['Game Name'] == game_name].index[0]

    distances = similarity[game_index]

    game_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:15]

    recommended_games = []

    for i in game_list:

        game_data = games.iloc[i[0]]

        # Weighted Ranking Formula

        similarity_score = i[1] * 100

        rating_score = game_data['Rating'] * 10

        popularity_score = game_data['Popularity Score']

        final_score = (
            0.5 * similarity_score +
            0.3 * rating_score +
            0.2 * popularity_score
        )

        # User Preference Boost

        if game_data['Platform'] == preferred_platform:
            final_score += 5

        if game_data['Mode'] == preferred_mode:
            final_score += 5

        if game_data['Rating'] >= minimum_rating:

            recommended_games.append({
                'Game': game_data['Game Name'],
                'Genre': game_data['Genre'],
                'Platform': game_data['Platform'],
                'Mode': game_data['Mode'],
                'Style': game_data['Style'],
                'Difficulty': game_data['Difficulty'],
                'Rating': game_data['Rating'],
                'Developer': game_data['Developer'],
                'Score': round(final_score, 2),

                'Reason': f"""
                Recommended because it shares similar gameplay,
                genre, and style with {selected_game}.
                It also matches your preferred platform and mode.
                """
            })

    recommendations = pd.DataFrame(recommended_games)

    recommendations = recommendations.sort_values(
        by='Score',
        ascending=False
    )

    return recommendations.head(5)

# ---------------- BUTTON ---------------- #

if st.button("🎮 Recommend Games"):

    results = recommend_games(selected_game)

    st.subheader("🔥 Top Recommended Games")

    # ---------------- GAME CARDS ---------------- #

    for i in range(len(results)):

        game = results.iloc[i]

        st.markdown(f"""
        <div style="
            background: linear-gradient(to right, #1E293B, #334155);
            padding: 22px;
            border-radius: 16px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        ">
        """, unsafe_allow_html=True)

        st.markdown(f"## 🎯 {game['Game']}")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Genre:** {game['Genre']}")
            st.write(f"**Platform:** {game['Platform']}")
            st.write(f"**Mode:** {game['Mode']}")
            st.write(f"**Difficulty:** {game['Difficulty']}")

        with col2:
            st.write(f"**Rating:** ⭐ {game['Rating']}")
            st.write(f"**Developer:** {game['Developer']}")
            st.write(f"**Recommendation Score:** 🔥 {game['Score']}")

        st.info(game['Reason'])

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TABLE ---------------- #

    st.subheader("📊 Recommendation Scores")

    chart_data = pd.DataFrame({
        'Game': results['Game'],
        'Recommendation Score': results['Score']
    })

    st.dataframe(chart_data, use_container_width=True)

    # ---------------- PROFESSIONAL VISUALIZATION ---------------- #

    st.subheader("📈 Recommendation Visualization")

    visual_df = pd.DataFrame({
        "Game": results['Game'],
        "Recommendation Score": results['Score']
    })

    # Sort values for better visualization
    visual_df = visual_df.sort_values(
        by="Recommendation Score",
        ascending=True
    )

    fig = px.bar(
        visual_df,
        x="Recommendation Score",
        y="Game",
        orientation='h',
        text="Recommendation Score",
        height=450
    )

    # Modern Styling
    fig.update_traces(
        textposition='outside',
        marker_color='#60A5FA'
    )

    fig.update_layout(
        paper_bgcolor='#0B1120',
        plot_bgcolor='#111827',
        font_color='white',
        xaxis_title='Recommendation Score',
        yaxis_title='Games',
        title='Top Recommended Games',
        title_x=0.3,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)