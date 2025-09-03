import streamlit as st
from datetime import datetime
import pandas as pd
from supabase_client import supabase

#st.set_page_config(page_title="Bike reservation", page_icon="📝", layout="centered")
st.header("Fill the form to reserve a bike")

# --- 1. Load bikes and stations from Supabase and convert to dataframe ---
bikes_res = supabase.table("bikes").select("*").execute()
bikes = pd.DataFrame(bikes_res.data)

stations_res = supabase.table("stations").select("*").order("NAME").execute()
stations = pd.DataFrame(stations_res.data)

# Distinct models (ordered)
models = sorted(bikes["MODEL"].dropna().unique().tolist())

# Build {name: station_id} dictionary for selectbox
stations_dict = dict(zip(stations["NAME"], stations["STATION_ID"]))
station_names = list(stations_dict.keys())

# --- 2. Reservation form ---
with st.form("reservation_form"):
    if not models:
        st.info("No bike model in database. Check the database.")
    model = st.selectbox("Modèle de vélo", models, disabled=not models)

    date_ret = st.date_input("Rent date")
    heure_ret = st.time_input("Rent time")
    start_station = st.selectbox("Pickup station", station_names, disabled=not station_names)
    end_station = st.selectbox("Return station", station_names, disabled=not station_names)
    duration = st.number_input("Duration (hours)", min_value=1, max_value=240, step=1, value=2)
    email = st.text_input("Email (UNIL/EPFL)", placeholder="prenom.nom@epfl.ch")

    submitted = st.form_submit_button("Reserve a bike")

# --- 3. Process reservation ---
# Processing on data after the form is submitted
if submitted: 
    # Check for requires fields
    if not model or not email or not start_station or not end_station:
        st.error("All fields are required, check the form and try again.")
        st.stop()

    # Pickup time with apropriate format
    pickup_ts = datetime.combine(date_ret, heure_ret)
    pickup_iso = pickup_ts.isoformat(sep=" ")

    # IDs des stations
    start_station_id = stations_dict[start_station]
    end_station_id = stations_dict[end_station]

    # 1) Find if a bike is available
    # It finds the first available bike of the selected model at the selected start station.
    bike_res = (
        supabase.table("bikes")
        .select("BIKE_ID")
        .eq("MODEL", model)
        .eq("STATUS", "AVAILABLE")
        .eq("STATION_ID", start_station_id)
        .limit(1)
        .execute()
    )

    if not bike_res.data:
        st.error("No available bike found for this model at the selected station. Try later or find another model/station in the \"Availability\" page.")
        st.stop()

    # Get the bike ID
    bike_id = bike_res.data[0]["BIKE_ID"]

    # 2) Insert reservation into "reservations" table
    supabase.table("reservations").insert({
        "BIKE_ID": bike_id,
        "START_STATION_ID": start_station_id,
        "END_STATION_ID": end_station_id,
        "USER_EMAIL": email,
        "PICKUP_TS": pickup_iso,
        "DURATION_HOURS": int(duration),
        "STATUS": "ACTIVE"
    }).execute()

    # 3) Update bike status to "RESERVED"
    supabase.table("bikes").update({"STATUS": "RESERVED"}).eq("BIKE_ID", bike_id).execute()

    # 4) Messages Streamlit
    st.success(f"Your booking is confirmed! \n Bike ID {bike_id} ({model}) le {pickup_iso} pour {int(duration)}h.")
    st.info(f"📍 Pickup: {start_station} → Return: {end_station}")
    st.write("Enjoy your ride!")
    st.balloons()