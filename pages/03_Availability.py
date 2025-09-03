import streamlit as st
import pandas as pd
from supabase_client import supabase

#st.set_page_config(page_title="Table of bike availability", page_icon="📊", layout="wide")
st.header("📊 Bike availability by station and model")

# 1. Function that loads bikes data with station information from Supabase
def load_bikes_df():
    try:
        # Query bikes with joined station data
        response = (
            supabase.table("bikes")
            .select(
                "BIKE_ID,MODEL,STATUS,STATION_ID,"
                "stations(NAME)" # uses the secondary key relationship
            )
            .order("stations(NAME),MODEL,BIKE_ID")
            .execute()
        )
        
        if not response.data:
            # No data found - stops the processing here and returns empty dataframe
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
        st.error(f"Error while loading data: {str(e)}")
        return pd.DataFrame()


# Optional : Cache the data loading function for better performance (avoid loading on every interaction)
@st.cache_data(ttl=60)  # Cache for 60 seconds
def get_bikes_data():
    return load_bikes_df()

# Optional : Manual refresh button
if st.button("Refresh now"):
    st.cache_data.clear()  # Clear cache on manual refresh
    st.rerun()

#2. Loads and display data with filters
# Load data
df = get_bikes_data()

# Filters definition on STATUS, STATION, MODEL.
# Show the three filter options in three columns
c1, c2, c3 = st.columns(3)
with c1:
    status_opts = df["STATUS"].dropna().unique().tolist()
    status_sel = st.multiselect("State", options=status_opts, default=status_opts)
with c2:
    station_opts = df["STATION"].dropna().unique().tolist()
    station_sel = st.multiselect("Station", options=station_opts, default=station_opts)
with c3:
    model_opts = df["MODEL"].dropna().unique().tolist()
    model_sel = st.multiselect("Model", options=model_opts, default=model_opts)

# Apply filters
mask = (
    df["STATUS"].isin(status_sel) &
    df["STATION"].isin(station_sel) &
    df["MODEL"].isin(model_sel)
)
view = df.loc[mask].reset_index(drop=True)

# Display filtered data in the app
st.dataframe(view, use_container_width=True, hide_index=True)

st.subheader("Information on the whole bike fleet")
# Additional metrics displayed in three columns
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total bikes", int(df.shape[0]))
with m2:
    st.metric("Available", int((df["STATUS"] == "AVAILABLE").sum()))
with m3:
    st.metric("Reserved", int((df["STATUS"] == "RESERVED").sum()))