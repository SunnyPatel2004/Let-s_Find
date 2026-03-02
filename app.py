# (Modified only for responsiveness — layout + mobile compatibility)
import streamlit as st
import pandas as pd
import numpy as np
import os
import gdown
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# PAGE CONFIG (sidebar collapsed improves mobile UX)
st.set_page_config(
    page_title="College Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# GLOBAL STYLE (RESPONSIVE IMPROVEMENTS ONLY)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #dbeafe, #eff6ff, #e0f2fe);
}
.block-container {
    padding: 1rem 3vw 2rem 3vw;
    max-width: 1200px;
}

/* Mobile layout */
@media (max-width: 768px) {
    .block-container { padding: 0.8rem 0.8rem 2rem 0.8rem; }
    .hero { padding: 40px 20px !important; }
    .hero-title-text { font-size: 30px !important; }
    .hero-sub-text { font-size: 14px !important; }
    .result-card { padding: 16px !important; }
    [data-testid="column"] { width: 100% !important; flex: 100% !important; }
}

.hero {
    background: linear-gradient(120deg, #2563eb, #0ea5e9);
    padding: 70px 60px;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0px 12px 30px rgba(37,99,235,0.25);
}
.hero-title-text {
    font-size: 46px;
    font-weight: 900;
    background: linear-gradient(90deg, #ffffff, #dbeafe, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub-text { margin-top: 10px; font-size: 18px; color: #e0f2fe; }

.result-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 6px 18px rgba(37,99,235,0.12);
    margin-bottom: 14px;
    border-left: 6px solid #0ea5e9;
}

.profile-section-title {
    font-size: 28px;
    font-weight: 900;
    margin-top: 30px;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.review-box {
    background: linear-gradient(90deg, #f0f9ff, #e0f2fe);
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 10px;
    border-left: 5px solid #3b82f6;
    white-space: pre-wrap;
    line-height: 1.6;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
    <div class="hero-title-text">Discover Your Perfect College</div>
    <div class="hero-sub-text">Place Where Dream Becomes Reality....</div>
</div>
""", unsafe_allow_html=True)

# LOAD DATA
CSV_PATH = "recommendation.csv"
NPY_PATH = "review_embeddings.npy"
CSV_URL = "https://drive.google.com/uc?id=1vnOUt5f6tveyejrgjBvFlw_2LFrtIFGb"
NPY_URL = "https://drive.google.com/uc?id=1g2bdBPvRCyx6CS3BoLwIPVt7ofjEeuoi"

def download_files():
    if not os.path.exists(CSV_PATH): gdown.download(CSV_URL, CSV_PATH, quiet=False)
    if not os.path.exists(NPY_PATH): gdown.download(NPY_URL, NPY_PATH, quiet=False)

@st.cache_data
def load_data():
    download_files()
    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(NPY_PATH)
    return df, embeddings

df, review_embeddings = load_data()

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# SEMANTIC FILTER
def semantic_filter(df_subset, query, threshold=0.35):
    if df_subset.empty: return df_subset
    query_embedding = model.encode([query])
    subset_embeddings = review_embeddings[df_subset.index]
    similarities = cosine_similarity(query_embedding, subset_embeddings)[0]
    df_subset = df_subset.copy()
    df_subset["Similarity"] = similarities
    return df_subset[df_subset["Similarity"] >= threshold]

# AGGREGATION
def aggregate_college_scores(filtered_df):
    college_scores = []
    for college, group in filtered_df.groupby("College"):
        review_count = len(group)
        if review_count == 0: continue
        weight = 1.0 if review_count>=30 else 0.75 if review_count>=10 else 0.5 if review_count>=5 else 0.25
        college_scores.append({
            "College": college,
            "Avg_Sentiment": group["Sentiment_Score"].mean()*weight,
            "Avg_Rating": group["Rating"].mean(),
            "Review_Count": review_count,
            "Class_Size": group["Class Size"].mean(),
            "Course_Fee": group["Course Fee"].mean()
        })
    return pd.DataFrame(college_scores)

# SIDEBAR
with st.sidebar:
    st.header("Filter Preferences")
    keyword = st.text_input("Keywords")
    degree_list = sorted(df["Degree"].dropna().unique())
    degree = st.selectbox("Degree", [""] + degree_list)
    spec_list = sorted(df[df["Degree"]==degree]["Specialization"].dropna().unique()) if degree else sorted(df["Specialization"].dropna().unique())
    specialization = st.selectbox("Specialization", [""] + spec_list)
    top_n = st.slider("Number of Recommendations", 1, 20, 10)
    search = st.button("Search Colleges")

# SEARCH
if search:
    filtered = df.copy()
    if degree: filtered = filtered[filtered["Degree"]==degree]
    if specialization: filtered = filtered[filtered["Specialization"]==specialization]
    if keyword: filtered = semantic_filter(filtered, keyword)
    st.session_state.results = aggregate_college_scores(filtered).sort_values(["Avg_Sentiment","Avg_Rating"],ascending=[False,False]).head(top_n)
    st.session_state.selected_college=None

# RESULTS
if "results" in st.session_state and st.session_state.results is not None:
    ranked=st.session_state.results.reset_index(drop=True); ranked.index+=1
    st.subheader("Top Recommendations")
    for rank,row in ranked.iterrows():
        st.markdown(f'<div class="result-card"><b>Rank #{rank} — {row["College"]}</b><br><br>⭐ Rating: {row["Avg_Rating"]:.2f} | 😊 Sentiment: {row["Avg_Sentiment"]:.3f} | 📝 Reviews Used: {int(row["Review_Count"])}</div>',unsafe_allow_html=True)
    st.markdown('<div class="profile-section-title">View Detailed Profile</div>',unsafe_allow_html=True)
    st.session_state.selected_college = st.selectbox("", ranked["College"].tolist(), label_visibility="collapsed")

# DETAILS
if "selected_college" in st.session_state and st.session_state.selected_college:
    college_name=st.session_state.selected_college
    college_df=df[df["College"]==college_name]
    if not college_df.empty:
        summary=aggregate_college_scores(college_df).iloc[0]
        st.header(college_name)
        c1,c2=st.columns(2,gap="large")
        c1.metric("Average Class Size", int(summary["Class_Size"]))
        c2.metric("Average Course Fee Annually", f"₹ {int(summary['Course_Fee']):,}")
        st.markdown("### Performance Overview")
        section_cols=["Campus Life","Faculty","Hostel Facilities","Placement Experience","Course Curriculum Overview"]
        available=[c for c in section_cols if c in college_df.columns]
        if available:
            avg_df=college_df[available].mean().reset_index(); avg_df.columns=["Category","Score"]; avg_df["Label"]=avg_df["Score"].map(lambda x:f"{x:.2f}")
            fig=px.bar(avg_df,x="Category",y="Score",text="Label",range_y=[0,5])
            fig.update_layout(height=420,margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,use_container_width=True,config={"responsive":True})
        st.markdown("### Reviews")
        for review in college_df["Review"].dropna().tolist()[:5]:
            short=review[:200]+"..." if len(review)>200 else review
            with st.expander(short): st.markdown(f'<div class="review-box">{review}</div>',unsafe_allow_html=True)
