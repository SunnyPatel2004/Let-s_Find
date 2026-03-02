import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
import plotly.express as px
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Detect Mobile Screen
# --------------------------------------------------
components.html("""
<script>
const width = window.innerWidth;
window.parent.postMessage({type: "streamlit:setComponentValue", value: width}, "*");
</script>
""", height=0)

screen_width = st.session_state.get("screen_width", 1200)
mobile = screen_width < 768


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="College Recommendation System", page_icon="🎓", layout="wide")


# --------------------------------------------------
# GLOBAL STYLE (Responsive)
# --------------------------------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #dbeafe, #eff6ff, #e0f2fe);
}

/* Container */
.block-container {
    padding: 1rem 4vw 2rem 4vw;
    max-width: 1200px;
}

@media (max-width: 768px) {
    .block-container {
        padding: 0.5rem 1rem 1.5rem 1rem;
    }
}

/* Hero */
.hero {
    background: linear-gradient(120deg, #2563eb, #0ea5e9);
    padding: clamp(25px, 6vw, 70px) clamp(20px, 5vw, 60px);
    border-radius: 20px;
    margin-bottom: 35px;
    box-shadow: 0px 12px 30px rgba(37,99,235,0.25);
}

.hero-title-text {
    font-size: clamp(26px, 6vw, 48px);
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff, #dbeafe, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub-text {
    margin-top: 12px;
    font-size: clamp(14px, 3.5vw, 18px);
    color: #e0f2fe;
}

/* Cards */
.result-card {
    background: white;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 5px 16px rgba(37,99,235,0.12);
    margin-bottom: 14px;
    border-left: 6px solid #0ea5e9;
}

/* Review */
.review-box {
    background: linear-gradient(90deg, #f0f9ff, #e0f2fe);
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #3b82f6;
    font-size: clamp(13px, 3.5vw, 14px);
    line-height: 1.7;
    overflow-wrap: anywhere;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HERO
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title-text">Discover Your Perfect College</div>
    <div class="hero-sub-text">Place Where Dream Becomes Reality....</div>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA DOWNLOAD
# --------------------------------------------------
CSV_PATH = "recommendation.csv"
NPY_PATH = "review_embeddings.npy"

CSV_URL = "https://drive.google.com/uc?id=1vnOUt5f6tveyejrgjBvFlw_2LFrtIFGb"
NPY_URL = "https://drive.google.com/uc?id=1g2bdBPvRCyx6CS3BoLwIPVt7ofjEeuoi"

def download_files():
    if not os.path.exists(CSV_PATH):
        gdown.download(CSV_URL, CSV_PATH, quiet=False)
    if not os.path.exists(NPY_PATH):
        gdown.download(NPY_URL, NPY_PATH, quiet=False)

@st.cache_data
def load_data():
    download_files()
    return pd.read_csv(CSV_PATH), np.load(NPY_PATH)

df, review_embeddings = load_data()

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# --------------------------------------------------
# SEMANTIC FILTER
# --------------------------------------------------
def semantic_filter(df_subset, query, threshold=0.35):
    if df_subset.empty:
        return df_subset
    query_embedding = model.encode([query])
    subset_embeddings = review_embeddings[df_subset.index]
    similarities = cosine_similarity(query_embedding, subset_embeddings)[0]
    df_subset = df_subset.copy()
    df_subset["Similarity"] = similarities
    return df_subset[df_subset["Similarity"] >= threshold]


# --------------------------------------------------
# AGGREGATION
# --------------------------------------------------
def aggregate_college_scores(filtered_df):
    results = []
    for college, group in filtered_df.groupby("College"):
        count = len(group)
        if count == 0:
            continue

        weight = 1 if count>=30 else 0.75 if count>=10 else 0.5 if count>=5 else 0.25

        results.append({
            "College": college,
            "Avg_Sentiment": group["Sentiment_Score"].mean()*weight,
            "Avg_Rating": group["Rating"].mean(),
            "Review_Count": count,
            "Class_Size": group["Class Size"].mean(),
            "Course_Fee": group["Course Fee"].mean()
        })
    return pd.DataFrame(results)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Filter Preferences")

keyword = st.sidebar.text_input("Keywords")
degree = st.sidebar.selectbox("Degree", [""] + sorted(df["Degree"].dropna().unique()))
specialization = st.sidebar.selectbox("Specialization", [""] + sorted(df["Specialization"].dropna().unique()))
top_n = st.sidebar.slider("Recommendations",1,20,10)
search = st.sidebar.button("Search Colleges")


# --------------------------------------------------
# SEARCH
# --------------------------------------------------
if search:
    filtered=df.copy()
    if degree: filtered=filtered[filtered["Degree"]==degree]
    if specialization: filtered=filtered[filtered["Specialization"]==specialization]
    if keyword: filtered=semantic_filter(filtered,keyword)

    st.session_state.results=aggregate_college_scores(filtered).sort_values(
        ["Avg_Sentiment","Avg_Rating"],ascending=[False,False]
    ).head(top_n)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------
if "results" in st.session_state:
    ranked=st.session_state.results.reset_index(drop=True)
    ranked.index+=1

    st.subheader("Top Recommendations")

    for r,row in ranked.iterrows():
        st.markdown(f"""
        <div class="result-card">
        <b>Rank #{r} — {row['College']}</b><br>
        ⭐ {row['Avg_Rating']:.2f} | 😊 {row['Avg_Sentiment']:.3f} | 📝 {int(row['Review_Count'])}
        </div>
        """,unsafe_allow_html=True)

    selected=st.selectbox("View College",ranked["College"])
    college_df=df[df["College"]==selected]


    # ---------------- METRICS ----------------
    if mobile:
        st.metric("Average Class Size",int(college_df["Class Size"].mean()))
        st.metric("Course Fee",f"₹ {int(college_df['Course Fee'].mean()):,}")
    else:
        c1,c2=st.columns(2)
        c1.metric("Average Class Size",int(college_df["Class Size"].mean()))
        c2.metric("Course Fee",f"₹ {int(college_df['Course Fee'].mean()):,}")


    # ---------------- CHART ----------------
    cols=["Campus Life","Faculty","Hostel Facilities","Placement Experience","Course Curriculum Overview"]
    avg_df=college_df[cols].mean().reset_index()
    avg_df.columns=["Category","Score"]

    fig=px.bar(avg_df,x="Category",y="Score",text=avg_df["Score"].round(2))
    fig.update_layout(
        height=350 if mobile else 420,
        margin=dict(l=10,r=10,t=30,b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title=None,yaxis_title="Rating"
    )
    fig.update_traces(textposition="outside")

    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})


    # ---------------- REVIEWS ----------------
    st.markdown("### Reviews")
    for review in college_df["Review"].dropna().head(5):
        with st.expander(review[:120]+"..."):
            st.markdown(f'<div class="review-box">{review}</div>',unsafe_allow_html=True)
