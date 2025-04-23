import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types

st.set_page_config(layout="wide")
st.title("📋 Insurance Preferences & Policy Renewal Analysis")

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("combined_insurance_data.xlsx")

df = load_data()

# Gemini-based LLM function
def generate_llm_insight(prompt):
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction="You are a data analyst summarizing insurance user survey data."),
        contents=[prompt],
    )
    return response.text.strip()

# Filters
city_col, age_col, edu_col, job_col = "City", "Age Group", "Education", "Occupation"
with st.sidebar:
    st.header("📋 Filters")
    selected_city = st.selectbox("City", ["All"] + sorted(df[city_col].dropna().unique()))
    selected_age = st.selectbox("Age Group", ["All"] + sorted(df[age_col].dropna().unique()))
    selected_edu = st.selectbox("Education", ["All"] + sorted(df[edu_col].dropna().unique()))
    selected_job = st.selectbox("Occupation", ["All"] + sorted(df[job_col].dropna().unique()))

for col, selected in zip([city_col, age_col, edu_col, job_col], [selected_city, selected_age, selected_edu, selected_job]):
    if selected != "All":
        df = df[df[col] == selected]

selected_section = st.radio("📊 Select Section", ["📱 Digital Preferences", "🎯 Policy Renewal Reasons", "📶 Digital Readiness"], horizontal=True)

if selected_section == "📱 Digital Preferences":
    col1, col2 = st.columns(2)
    with col1:
        # st.subheader("📱 Digital Channel Preference")
        # Select columns with digital channels
        digital_cols = [col for col in df.columns if any(x in col for x in ["WhatsApp", "App", "Website", "Chatbot", "Call Centre"])]

        # Summarize values
        digital_summary = df[digital_cols].apply(pd.to_numeric, errors='coerce').sum().reset_index()
        digital_summary.columns = ["Channel", "Count"]

        # Sort the data for better visualization
        digital_summary = digital_summary.sort_values(by="Count", ascending=False)

        # Split by label length
        short_labels = digital_summary[digital_summary["Channel"].str.len() < 30]

        # Plot 1: Short labels
        fig1 = px.bar(short_labels, x="Channel", y="Count", title="Digital Channel Preference ", color="Channel")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("🧠 Gemini Insights")
        digital_prompt = f"""
        Based on this digital preference data (total counts):
        {digital_summary.to_string(index=False)}
        Summarize 2–3 key insights in simple language.
        """
        insight = generate_llm_insight(digital_prompt)
        st.markdown(f"**Insight:**\n{insight}")

elif selected_section == "🎯 Policy Renewal Reasons":
    col1, col2 = st.columns(2)
    with col1:
        # st.subheader("🎯 Policy Renewal Reasons (Weighted)")
        renew_cols = [col for col in df.columns if "renew the policy" in col]
        weights = {"Rank 1": 4, "Rank 2": 3, "Rank 3": 2, "Rank 4": 1}
        weighted_scores = {}
        for col in renew_cols:
            rank_label = [k for k in weights if k in col][0]
            for val in df[col].dropna():
                reason = str(val).strip()
                weighted_scores[reason] = weighted_scores.get(reason, 0) + weights[rank_label]
        renewal_scores = pd.DataFrame(weighted_scores.items(), columns=["Reason", "Score"]).sort_values(by="Score")
        fig = px.bar(renewal_scores, x="Score", y="Reason", orientation='h', title="Top Policy Renewal Reasons", color="Score", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🧠 Gemini Insights")
        renew_prompt = f"""
        Based on the following policy renewal reason ranking scores:
        {renewal_scores.to_string(index=False)}
        Summarize the key reasons users renew their policies.
        """
        insight = generate_llm_insight(renew_prompt)
        st.markdown(f"**Insight:**\n{insight}")

elif selected_section == "📶 Digital Readiness":
    col1, col2 = st.columns(2)
    with col1:
        # st.subheader("📶 Digital Readiness Score by Age")
        digital_cols = [col for col in df.columns if any(x in col for x in ["WhatsApp", "App", "Website", "Chatbot", "Call Centre"])]
        df["digital_score"] = df[digital_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        readiness_scores = df.groupby(age_col)["digital_score"].mean().sort_values().reset_index()
        fig = px.bar(readiness_scores, x=age_col, y="digital_score", title="Average Digital Readiness Score by Age", color="digital_score", color_continuous_scale="Purples")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🧠 Gemini Insights")
        readiness_prompt = f"""
        Based on the average digital readiness score by age group:
        {readiness_scores.to_string(index=False)}
        Provide 2-3 insights about digital channel adoption.
        """
        insight = generate_llm_insight(readiness_prompt)
        st.markdown(f"**Insight:**\n{insight}")
