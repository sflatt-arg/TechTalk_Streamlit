# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session

# Write directly to the app
st.set_page_config(page_title="Publibike/Velospot", initial_sidebar_state="expanded")

st.title("Publibike / Velospot — Démo Streamlit (UNIL–EPFL)")
st.markdown("""
This app is made of four navigation pages:
1. **Bike reservation**   
2. **Bike return**  
3. **Availabilities in bike stations**
4. **Map**
""")

page_reservation = st.Page("pages/01_Reservation.py", title="Reservation")
page_return = st.Page("pages/04_Return.py", title="Return")
page_availability = st.Page("pages/02_Availability.py", title="Bike availability")
page_map = st.Page("pages/03_Map.py", title="Map")

pages = {
    "Rental": [page_reservation, page_return],
    "Bikes and stations" : [page_availability, page_map],
}

pg = st.navigation(pages, position="top")
pg.run()