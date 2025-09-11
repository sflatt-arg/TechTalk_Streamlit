from openai import OpenAI
import streamlit as st

# Load your API key from st.secrets (or environment variable)
openai_client = OpenAI(api_key=st.secrets["openai"]["api_key"])
