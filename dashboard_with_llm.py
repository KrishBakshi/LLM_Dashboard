import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from google import genai
from google.genai import types
from access_token import GOOGLE_API_KEY


# Gemini-based LLM function
def generate_llm_insight(prompt):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction="You are a data analyst summarizing insurance user survey data."),
        contents=[prompt],
    )
    return response.text.strip()

# Load cleaned preprocessed data
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_data.csv")
    return df

# Plot top insurers
def plot_insurer_counts(df):
    insurer_col = [col for col in df.columns if col.endswith("S7")][0]
    top_insurers = df[insurer_col].value_counts().head(10)
    fig, ax = plt.subplots()
    top_insurers.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title("Top 10 Insurers by Respondent Count")
    ax.set_xlabel("Number of Respondents")
    ax.invert_yaxis()
    st.pyplot(fig)

# Plot digital channel preferences
def plot_digital_preferences(df):
    digital_cols = [col for col in df.columns if "Q16_AO" in col]
    digital_summary = df[digital_cols].apply(pd.to_numeric, errors='coerce').sum()
    fig, ax = plt.subplots()
    digital_summary.plot(kind='bar', color='mediumseagreen', ax=ax)
    ax.set_title("Digital Channel Preference")
    ax.set_ylabel("Number of Respondents")
    st.pyplot(fig)
    return digital_summary

# Plot renewal reason weighted scores
def plot_renewal_reasons(df):
    renew_cols = [col for col in df.columns if "Q15_" in col]
    weights = {"Q15_1": 4, "Q15_2": 3, "Q15_3": 2, "Q15_4": 1}

    weighted_scores = {}
    for col in renew_cols:
        rank_weight = weights.get(col.split("_")[1], 0)
        for val in df[col].dropna():
            reason = str(val).strip()
            weighted_scores[reason] = weighted_scores.get(reason, 0) + rank_weight

    sorted_scores = pd.Series(weighted_scores).sort_values(ascending=False)
    fig, ax = plt.subplots()
    sorted_scores.plot(kind='barh', color='coral', ax=ax)
    ax.set_title("Top Policy Renewal Reasons (Weighted by Rank)")
    ax.set_xlabel("Total Weighted Score")
    ax.invert_yaxis()
    st.pyplot(fig)
    return sorted_scores

# Streamlit layout
st.set_page_config(layout="wide")
st.title("🧠 Insurance Data Analysis Dashboard with Gemini Insights")

df = load_data()

# DEMOGRAPHIC FILTERS
city_col = [col for col in df.columns if col.endswith("S4")][0]
age_col = [col for col in df.columns if col.endswith("S5")][0]
edu_col = [col for col in df.columns if col.endswith("S6a")][0]
job_col = [col for col in df.columns if col.endswith("S6b")][0]

with st.sidebar:
    st.header("📋 Filters")
    selected_city = st.selectbox("City", options=["All"] + sorted(df[city_col].dropna().unique().tolist()))
    selected_age = st.selectbox("Age Group", options=["All"] + sorted(df[age_col].dropna().unique().tolist()))
    selected_edu = st.selectbox("Education", options=["All"] + sorted(df[edu_col].dropna().unique().tolist()))
    selected_job = st.selectbox("Employment", options=["All"] + sorted(df[job_col].dropna().unique().tolist()))

# Apply filters
if selected_city != "All":
    df = df[df[city_col] == selected_city]
if selected_age != "All":
    df = df[df[age_col] == selected_age]
if selected_edu != "All":
    df = df[df[edu_col] == selected_edu]
if selected_job != "All":
    df = df[df[job_col] == selected_job]

# LAYOUT
col1, col2 = st.columns([1.5, 1.5])

with col1:
    # st.subheader("Top Insurers")
    with st.container():
        st.markdown("""
                    <h4 style='margin-bottom: 0;'>Top Insurers</h4>
                    """, unsafe_allow_html=True)
        # chart
        plot_insurer_counts(df)

    st.subheader("Digital Channel Preferences")
    digital_summary = plot_digital_preferences(df)

    st.subheader("Top Policy Renewal Reasons")
    renewal_scores = plot_renewal_reasons(df)

with col2:
    st.subheader("🔍 AI Insight")
    digital_prompt = f"""
    Based on this digital preference data (total counts):
    {digital_summary.to_string()}

    Please summarize 2–3 key insights in simple language.
    """
    insight = generate_llm_insight(digital_prompt)
    st.markdown(f"**Insight:**\n{insight}")

    suggestion_prompt = f"""
    Based on this insight:
    {insight}

    Suggest one action to improve digital engagement for renewals.
    """
    suggestion = generate_llm_insight(suggestion_prompt)
    st.subheader("💡 Suggestion")
    st.markdown(f"**Suggestion:**\n{suggestion}")
