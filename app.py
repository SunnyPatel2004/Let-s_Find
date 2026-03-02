import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# PAGE CONFIG
st.set_page_config(
    page_title="College Recommendation System",
    page_icon="🎓",
    layout="wide"
)

# GLOBAL STYLE (UNCHANGED UI)
st.markdown("""
<style>

/* PAGE BACKGROUND */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #dbeafe, #eff6ff, #e0f2fe);
}

.block-container {
    padding-top: 1rem;
    padding-left: 5rem;
    padding-right: 5rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

.hero {
    background: linear-gradient(120deg, #2563eb, #0ea5e9);
    padding: 70px 60px;
    border-radius: 20px;
    margin-bottom: 40px;
    box-shadow: 0px 12px 30px rgba(37,99,235,0.25);
}

.hero-title-text {
    font-size: 48px;
    font-weight: 900;
    letter-spacing: 1.2px;
    background: linear-gradient(90deg, #ffffff, #dbeafe, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub-text {
    margin-top: 15px;
    font-size: 18px;
    font-weight: 500;
    color: #e0f2fe;
}

.result-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(37,99,235,0.12);
    margin-bottom: 18px;
    border-left: 6px solid #0ea5e9;
}

.profile-section-title {
    font-size: 30px;
    font-weight: 900;
    margin-top: 35px;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.review-box {
    background: linear-gradient(90deg, #f0f9ff, #e0f2fe);
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
    border-left: 5px solid #3b82f6;
    white-space: pre-wrap;
    line-height: 1.7;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-title-text">
        Discover Your Perfect College
    </div>
    <div class="hero-sub-text">
        Place Where Dream Becomes Reality....
    </div>
</div>
""", unsafe_allow_html=True)


# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/patel/Desktop/Python/UserInterface/Recommendation.csv")
    df = df.reset_index(drop=True)
    embeddings = np.load("C:/Users/patel/Desktop/Python/UserInterface/review_embeddings.npy")
    return df, embeddings

df, review_embeddings = load_data()


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# 🔹 SEMANTIC FILTER (REAL FILTER)
def semantic_filter(df_subset, query, threshold=0.35):
    if df_subset.empty:
        return df_subset

    query_embedding = model.encode([query])
    subset_embeddings = review_embeddings[df_subset.index]

    similarities = cosine_similarity(query_embedding, subset_embeddings)[0]

    df_subset = df_subset.copy()
    df_subset["Similarity"] = similarities

    return df_subset[df_subset["Similarity"] >= threshold]


# 🔹 INTELLIGENT AGGREGATION (RELIABILITY BASED)
def aggregate_college_scores(filtered_df):

    college_scores = []

    for college, group in filtered_df.groupby("College"):

        review_count = len(group)
        if review_count == 0:
            continue

        # Reliability Weight
        if review_count >= 10:
            weight = 1.0
        elif review_count >= 5:
            weight = 0.75
        elif review_count >= 2:
            weight = 0.5
        else:
            weight = 0.25

        avg_sentiment = group["Sentiment_Score"].mean()
        avg_rating = group["Rating"].mean()

        adjusted_sentiment = avg_sentiment * weight

        college_scores.append({
            "College": college,
            "Avg_Sentiment": adjusted_sentiment,
            "Avg_Rating": avg_rating,
            "Review_Count": review_count,
            "Class_Size": group["Class Size"].mean(),
            "Course_Fee": group["Course Fee"].mean()
        })

    return pd.DataFrame(college_scores)


# SIDEBAR
st.sidebar.header("Filter Preferences")

keyword = st.sidebar.text_input("Keywords (Good College,High Placement,Good Faculty)")
degree_list = sorted(df["Degree"].dropna().unique())
degree = st.sidebar.selectbox("Degree", [""] + degree_list)

if degree:
    spec_list = sorted(df[df["Degree"] == degree]["Specialization"].dropna().unique())
else:
    spec_list = sorted(df["Specialization"].dropna().unique())

specialization = st.sidebar.selectbox("Specialization", [""] + spec_list)

top_n = st.sidebar.slider("Number of Recommendations", 1, 20, 10)
search = st.sidebar.button("Search Colleges")


# SEARCH LOGIC (UPDATED)
if search:

    filtered = df.copy()

    if degree:
        filtered = filtered[filtered["Degree"] == degree]

    if specialization:
        filtered = filtered[filtered["Specialization"] == specialization]

    if keyword:
        filtered = semantic_filter(filtered, keyword)

    aggregated = aggregate_college_scores(filtered)

    st.session_state.results = aggregated.sort_values(
        ["Avg_Sentiment","Avg_Rating"],
        ascending=[False, False]
    ).head(top_n)

    st.session_state.selected_college = None


# RESULTS DISPLAY
if "results" in st.session_state and st.session_state.results is not None:

    ranked = st.session_state.results.reset_index(drop=True)
    ranked.index += 1

    st.subheader("Top Recommendations")

    for rank, row in ranked.iterrows():
        st.markdown(f"""
        <div class="result-card">
            <b>Rank #{rank} — {row['College']}</b><br><br>
            ⭐ Rating: {row['Avg_Rating']:.2f} |
            😊 Sentiment: {row['Avg_Sentiment']:.3f} |
            📝 Reviews Used: {int(row['Review_Count'])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="profile-section-title">View Detailed Profile</div>', unsafe_allow_html=True)

    selected = st.selectbox("", ranked["College"].tolist(), label_visibility="collapsed")
    st.session_state.selected_college = selected


# COLLEGE DETAILS
if "selected_college" in st.session_state and st.session_state.selected_college:

    college_name = st.session_state.selected_college
    college_df = df[df["College"] == college_name]

    if not college_df.empty:

        summary = aggregate_college_scores(college_df).iloc[0]

        st.header(college_name)

        col1, col2 = st.columns(2)
        col1.metric("Average Class Size", int(summary["Class_Size"]))
        col2.metric("Average Course Fee Annually", f"₹ {int(summary['Course_Fee']):,}")

        st.markdown("### Performance Overview")

        section_cols = [
            "Campus Life",
            "Faculty",
            "Hostel Facilities",
            "Placement Experience",
            "Course Curriculum Overview"
        ]

        available = [c for c in section_cols if c in college_df.columns]

        if available:
            avg_df = college_df[available].mean().reset_index()
            avg_df.columns = ["Category", "Score"]

            fig = px.bar(avg_df, x="Category", y="Score", text="Score", range_y=[0,5])
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Aspects",
                yaxis_title="Score (Out of 5)",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Reviews")

        reviews = college_df["Review"].dropna().tolist()
        for review in reviews[:5]:
            short = review[:200] + "..." if len(review) > 200 else review
            with st.expander(short):
                st.markdown(f'<div class="review-box">{review}</div>', unsafe_allow_html=True)
