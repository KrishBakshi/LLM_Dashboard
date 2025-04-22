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

@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_data.csv")
    return df

def plot_digital_readiness(df):
    digital_cols = [col for col in df.columns if "Q16_AO" in col]
    age_col = [col for col in df.columns if col.endswith("S5")][0]
    df["digital_score"] = df[digital_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    readiness = df.groupby(age_col)["digital_score"].mean().sort_values()

    fig, ax = plt.subplots()
    readiness.plot(kind='barh', color='purple', ax=ax)
    ax.set_title("Average Digital Readiness Score by Age Group")
    ax.set_xlabel("Digital Score")
    st.pyplot(fig)
    return readiness

st.set_page_config(layout="wide")
st.title("🧠 Insurance Data Analysis Dashboard with Gemini Insights")

df = load_data()

city_col = [col for col in df.columns if col.endswith("S4")][0]
age_col = [col for col in df.columns if col.endswith("S5")][0]
edu_col = [col for col in df.columns if col.endswith("S6a")][0]
job_col = [col for col in df.columns if col.endswith("S6b")][0]

with st.sidebar:
    st.header("📋 Filters")
    selected_city = st.selectbox("City", ["All"] + sorted(df[city_col].dropna().unique()))
    selected_age = st.selectbox("Age Group", ["All"] + sorted(df[age_col].dropna().unique()))
    selected_edu = st.selectbox("Education", ["All"] + sorted(df[edu_col].dropna().unique()))
    selected_job = st.selectbox("Employment", ["All"] + sorted(df[job_col].dropna().unique()))

if selected_city != "All":
    df = df[df[city_col] == selected_city]
if selected_age != "All":
    df = df[df[age_col] == selected_age]
if selected_edu != "All":
    df = df[df[edu_col] == selected_edu]
if selected_job != "All":
    df = df[df[job_col] == selected_job]

selected_section = st.radio("📊 Select Section", ["Digital Preferences", "Policy Renewal Reasons", "Digital Readiness"], horizontal=True)

if selected_section == "Digital Preferences":
    col1, col2 = st.columns([1.5, 1.5])
    with col1:
        st.subheader("📱 Digital Channel Preference")
        digital_cols = [col for col in df.columns if "Q16_AO" in col]
        digital_summary = df[digital_cols].apply(pd.to_numeric, errors='coerce').sum()
        fig, ax = plt.subplots()
        digital_summary.plot(kind='bar', color='mediumseagreen', ax=ax)
        ax.set_title("Digital Channel Preference")
        ax.set_ylabel("Number of Respondents")
        st.pyplot(fig)
    with col2:
        st.subheader("🧠 Gemini Insights")
        digital_prompt = f"""
        Based on this digital preference data (total counts):
        {digital_summary.to_string()}
        Summarize 2–3 key insights in simple language.
        """
        insight = generate_llm_insight(digital_prompt)
        st.markdown(f"**Insight:**\n{insight}")

elif selected_section == "Policy Renewal Reasons":
    col1, col2 = st.columns([1.5, 1.5])
    with col1:
        st.subheader("🎯 Policy Renewal Reasons (Weighted)")
        renew_cols = [col for col in df.columns if "Q15_" in col]
        weights = {"Q15_1": 4, "Q15_2": 3, "Q15_3": 2, "Q15_4": 1}
        weighted_scores = {}
        for col in renew_cols:
            rank_weight = weights.get(col.split("_")[1], 0)
            for val in df[col].dropna():
                reason = str(val).strip()
                weighted_scores[reason] = weighted_scores.get(reason, 0) + rank_weight
        renewal_scores = pd.Series(weighted_scores).sort_values(ascending=False)
        fig, ax = plt.subplots()
        renewal_scores.plot(kind='barh', color='coral', ax=ax)
        ax.set_title("Top Policy Renewal Reasons (Weighted by Rank)")
        ax.set_xlabel("Total Weighted Score")
        ax.invert_yaxis()
        st.pyplot(fig)
    with col2:
        st.subheader("🧠 Gemini Insights")
        renew_prompt = f"""
        Based on the following policy renewal reason ranking scores:
        {renewal_scores.to_string()}
        Summarize the key reasons users renew their policies.
        """
        insight = generate_llm_insight(renew_prompt)
        st.markdown(f"**Insight:**\n{insight}")


elif selected_section == "Digital Readiness":
    col1, col2 = st.columns([1.5, 1.5])
    with col1:
        st.subheader("📶 Digital Readiness Score by Age")
        readiness_scores = plot_digital_readiness(df)
    with col2:
        st.subheader("🧠 Gemini Insights")
        readiness_prompt = f"""
        Based on the average digital readiness score by age group:
        {readiness_scores.to_string()}
        Provide 2-3 insights about digital channel adoption.
        """
        insight = generate_llm_insight(readiness_prompt)
        st.markdown(f"**Insight:**\n{insight}")

