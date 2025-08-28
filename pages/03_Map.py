import streamlit as st
import pydeck as pdk

st.header("🗺️ Carte des vélos disponibles à Lausanne")

session = get_active_session()

df = session.sql("""
    SELECT 
      s.NAME, s.LATITUDE, s.LONGITUDE,
      COUNT(*) AS NB_VELOS
    FROM BIKES b
    JOIN STATIONS s ON b.STATION_ID = s.STATION_ID
    WHERE b.STATUS = 'AVAILABLE'
    GROUP BY s.NAME, s.LATITUDE, s.LONGITUDE
""").to_pandas()

if df.empty:
    st.info("Aucun vélo disponible pour le moment.")
    st.stop()

# Vue centrée Lausanne (approx.)
view_state = pdk.ViewState(latitude=46.5191, longitude=6.6291, zoom=13)

# Create a copy to avoid modifying original
df_map = df.copy()

# Simple approach - inline color calculation
scatter = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position="[LONGITUDE, LATITUDE]",
    get_radius=140,
    get_fill_color=[50, 205, 50],  # Fixed green color first
    pickable=True
)

# Simple labels
labels = pdk.Layer(
    "TextLayer",
    data=df_map.assign(TEXT=df_map["NB_VELOS"].astype(str)),
    get_position="[LONGITUDE, LATITUDE]",
    get_text="TEXT",
    get_size=16,
    get_color=[255, 255, 255, 255],
    get_alignment_baseline="'center'"
)

# Basic tooltip
tooltip = {"html": "<b>{NAME}</b><br/>Vélos: {NB_VELOS}", "style": {"color": "white"}}

# Basic deck without custom map style
st.pydeck_chart(pdk.Deck(
    layers=[scatter, labels], 
    initial_view_state=view_state, 
    tooltip=tooltip,
    map_style='light'
))