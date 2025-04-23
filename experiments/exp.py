import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from google import genai
from google.genai import types
from access_token import GOOGLE_API_KEY

# LLM Integration for Insight Generation
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
    return pd.read_excel("./Preprocessed_Motor_Insurance_Data.xlsx")

df = load_data()

# Shared columns
city_col, age_col, edu_col, job_col = "City", "Age Group", "Education", "Occupation"

# Page 1: Demographics
if __name__ == "__main__" and st.session_state.get("page") == "demographics":
    st.set_page_config(layout="wide")
    st.title("👥 Insurance Demographics Overview")

    st.subheader("🧬 Respondent Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        age_dist = df[age_col].value_counts()
        plt.figure()
        age_dist.plot.pie(autopct='%1.1f%%')
        plt.title("Age Group Distribution")
        plt.ylabel("")
        st.pyplot(plt)
    with col2:
        job_dist = df[job_col].value_counts()
        job_dist.plot(kind='bar', color='skyblue')
        plt.title("Occupation Type")
        st.pyplot(plt)
    with col3:
        edu_dist = df[edu_col].value_counts()
        edu_dist.plot(kind='bar', color='coral')
        plt.title("Education Level")
        st.pyplot(plt)

# Page 2 is handled by your 2.py file

# Page 3: Cross Tabs
elif __name__ == "__main__" and st.session_state.get("page") == "cross_tabs":
    st.set_page_config(layout="wide")
    st.title("🔁 Cross-Tabulated Insights")

    st.subheader("📊 Group 1: Digital Preferences by Demographics")
    app_cross = pd.crosstab(df[age_col], df['Digital - App'])
    web_cross = pd.crosstab(df[age_col], df['Digital - Website'])
    edu_app_cross = pd.crosstab(df[edu_col], df['Digital - App'])

    st.markdown("**📱 App Usage by Age Group**")
    app_cross.plot(kind='bar', stacked=True)
    st.pyplot(plt)

    st.markdown("**🌐 Website Usage by Age Group**")
    web_cross.plot(kind='bar', stacked=True, colormap='magma')
    st.pyplot(plt)

    st.markdown("**📱 App Usage by Education Level**")
    edu_app_cross.plot(kind='bar', stacked=True, colormap='cividis')
    st.pyplot(plt)

    st.subheader("📊 Group 2: Trust & Claims")
    trust_cross = pd.crosstab(df[edu_col], df['Trust'])
    claim_cross = pd.crosstab(df['Claim Made'], df['Trust'])

    st.markdown("**🔐 Trust Level by Education**")
    trust_cross.plot(kind='bar', stacked=True, colormap='Greens')
    st.pyplot(plt)

    st.markdown("**📢 Claim History vs Trust**")
    claim_cross.plot(kind='bar', stacked=True, colormap='Oranges')
    st.pyplot(plt)

    st.subheader("📊 Group 3: Loyalty Signals")
    nps_cross = pd.crosstab(df[edu_col], df['NPS'])
    st.markdown("**🌟 Education Level vs Likelihood to Recommend (NPS)**")
    nps_cross.plot(kind='bar', stacked=True, colormap='coolwarm')
    st.pyplot(plt)

    with st.expander("🧠 Gemini-Generated Insights"):
        prompt = f"""
        Based on the following cross-tabulated values:
        App Usage by Age Group:
        {app_cross.to_string()}

        Website Usage by Age Group:
        {web_cross.to_string()}

        App Usage by Education:
        {edu_app_cross.to_string()}

        Education vs Trust:
        {trust_cross.to_string()}

        Claim Made vs Trust:
        {claim_cross.to_string()}

        Education vs NPS:
        {nps_cross.to_string()}

        Summarize the 3 most interesting findings for a business analyst.
        """
        insight = generate_llm_insight(prompt)
        st.markdown(f"**Insight:**\n{insight}")
