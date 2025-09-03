import streamlit as st
import pandas as pd
import pydeck as pdk
from supabase_client import supabase

st.header("🗺️ Map of bikes in Lausanne")

def load_available_bikes_map_data():
    """This function loads the available bikes with their station coordinates from Supabase.
    Arguments:
        None
    Returns:
         pd.DataFrame: Dataframe with columns: NAME, LATITUDE, LONGITUDE, N_BIKES
    """
    
    try:
        # Query bikes with station data, filtered for AVAILABLE bikes only
        response = (
            supabase.table("bikes")
            .select(
                "STATION_ID,"
                "stations(NAME,LATITUDE,LONGITUDE)"
            )
            .eq("STATUS", "AVAILABLE")
            .execute()
        )
        
        if not response.data:
            #returns empty dataframe if no data
            return pd.DataFrame()
        
        # Process and group data by station
        station_data = {}
        
        for bike in response.data:
            station = bike["stations"]
            if not station:  # Skip if station data is missing
                continue
                
            station_id = bike["STATION_ID"]
            station_name = station["NAME"]
            latitude = station["LATITUDE"]
            longitude = station["LONGITUDE"]
            
            # Group by station (using station_id as key to handle duplicate names)
            if station_id not in station_data:
                station_data[station_id] = {
                    "NAME": station_name,
                    "LATITUDE": float(latitude),
                    "LONGITUDE": float(longitude),
                    "N_BIKES": 0
                }
            
            station_data[station_id]["N_BIKES"] += 1
        
        # Convert to DataFrame
        rows = list(station_data.values())
        return pd.DataFrame(rows)
        
    except Exception as e:
        st.error(f"Error while loading data: {str(e)}")
        return pd.DataFrame()

# Optional Cache the data loading function for better performance
@st.cache_data(ttl=60)  # Cache for 60 seconds (longer for map data)
def get_map_data():
    return load_available_bikes_map_data()

# Load data using the cached function
df = get_map_data()

if df.empty:
    st.info("No available bike for the moment. Please check later.")
    st.stop()

# Centered view on Lausanne (approx.)
view_state = pdk.ViewState(latitude=46.5191, longitude=6.6291, zoom=13)

# Create a copy to avoid modifying original
df_map = df.copy()

# Create layers using pydeck
# Scatter layer for bike stations
scatter = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position="[LONGITUDE, LATITUDE]",
    get_radius=140,
    get_fill_color=[50, 205, 50],  # Fixed green color first
    pickable=True
)

# Simple labels with the number of available bikes
labels = pdk.Layer(
    "TextLayer",
    data=df_map.assign(TEXT=df_map["N_BIKES"].astype(str)),
    get_position="[LONGITUDE, LATITUDE]",
    get_text="TEXT",
    get_size=16,
    get_color=[255, 255, 255, 255],
    get_alignment_baseline="'center'"
)

# Simple tooltip with station name and number of bikes
tooltip = {"html": "<b>{NAME}</b><br/>Vélos: {N_BIKES}", "style": {"color": "white"}}

# Basic deck without custom map style
# Combine layers, labels and tooltip to display the map
st.pydeck_chart(pdk.Deck(
    layers=[scatter, labels], 
    initial_view_state=view_state, 
    tooltip=tooltip,
    map_style='light'
))

# Debug info (optional - remove in production)
with st.expander("Additional information"):
    st.dataframe(df, width='stretch')