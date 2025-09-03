# Import python packages
import streamlit as st

# Write directly to the app
st.set_page_config(page_title="Publibike/Velospot")

page_home = st.Page("pages/00_home.py", title="Home")
page_reservation = st.Page("pages/01_reservation.py", title="Reservation")
page_return = st.Page("pages/02_return.py", title="Return")
page_availability = st.Page("pages/03_availability.py", title="Bike availability")
page_map = st.Page("pages/04_map.py", title="Map")

pages = {
    "Home": [page_home],
    "Rental": [page_reservation,page_return],
    "Bikes and stations" : [page_availability,page_map],
}

pg = st.navigation(pages, position="top")
pg.run()
