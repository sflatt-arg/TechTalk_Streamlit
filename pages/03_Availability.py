import streamlit as st
import pandas as pd
from supabase import create_client

st.set_page_config(page_title="Parc vélo — Tableau", page_icon="📊", layout="wide")
st.header("📊 Parc vélo — Disponibles / Réservés / Maintenance")

# Load credentials from Streamlit secrets
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]
supabase = create_client(url, key)

def load_bikes_df():
    """Load bikes data with station information using Supabase"""
    try:
        # Query bikes with joined station data
        response = (
            supabase.table("bikes")
            .select(
                "BIKE_ID,MODEL,STATUS,STATION_ID,"
                "stations(NAME)"
            )
            .order("stations(NAME),MODEL,BIKE_ID")
            .execute()
        )
        
        if not response.data:
            return pd.DataFrame()
        
        # Convert to DataFrame and flatten the nested station data
        rows = []
        for bike in response.data:
            row = {
                "STATION": bike["stations"]["NAME"] if bike["stations"] else "Unknown",
                "MODEL": bike["MODEL"], 
                "STATUS": bike["STATUS"],
                "BIKE_ID": bike["BIKE_ID"],
                }
            rows.append(row)
        
        return pd.DataFrame(rows)
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return pd.DataFrame()

# Cache the data loading function for better performance
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_bikes_data():
    return load_bikes_df()

if st.button("Refresh now"):
    st.cache_data.clear()  # Clear cache on manual refresh
    st.rerun()

# Load data
df = get_bikes_data()

if df.empty:
    st.info("Pas de données. Initialise les tables.")
    st.stop()

# Filtres
c1, c2, c3 = st.columns(3)
with c1:
    status_opts = df["STATUS"].dropna().unique().tolist()
    status_sel = st.multiselect("État", options=status_opts, default=status_opts)
with c2:
    station_opts = df["STATION"].dropna().unique().tolist()
    station_sel = st.multiselect("Station", options=station_opts, default=station_opts)
with c3:
    model_opts = df["MODEL"].dropna().unique().tolist()
    model_sel = st.multiselect("Modèle", options=model_opts, default=model_opts)

# Apply filters
mask = (
    df["STATUS"].isin(status_sel) &
    df["STATION"].isin(station_sel) &
    df["MODEL"].isin(model_sel)
)
view = df.loc[mask].reset_index(drop=True)

# Display filtered data
st.dataframe(
    view,
    use_container_width=True,
    hide_index=True
)

# Petites métriques
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Vélos total", int(df.shape[0]))
with m2:
    st.metric("Disponibles", int((df["STATUS"] == "AVAILABLE").sum()))
with m3:
    st.metric("Réservés", int((df["STATUS"] == "RESERVED").sum()))