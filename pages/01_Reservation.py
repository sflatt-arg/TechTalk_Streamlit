import streamlit as st
from datetime import datetime
import pandas as pd
from supabase import create_client

# Load credentials from Streamlit secrets
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]

supabase = create_client(url, key)

st.set_page_config(page_title="Réservation de vélo", page_icon="📝", layout="centered")
st.header("📝 Réservation d’un vélo")



# Utilitaire simple pour échapper les quotes dans les strings SQL
def sql_escape(s: str) -> str:
    return (s or "").replace("'", "''")


# --- 2. Load bikes and stations from Supabase ---
bikes_res = supabase.table("bikes").select("*").execute()
stations_res = supabase.table("stations").select("*").order("NAME").execute()

bikes = pd.DataFrame(bikes_res.data)
stations = pd.DataFrame(stations_res.data)

# Distinct models (ordered)
models = sorted(bikes["MODEL"].dropna().unique().tolist())

# Build {name: station_id} dict
stations_dict = dict(zip(stations["NAME"], stations["STATION_ID"]))
station_names = list(stations_dict.keys())

# --- 3. Reservation form ---
with st.form("reservation_form"):
    if not models:
        st.info("No bike model in database. Check the database.")
    model = st.selectbox("Modèle de vélo", models, disabled=not models)

    date_ret = st.date_input("Date de retrait")
    heure_ret = st.time_input("Heure de retrait")
    start_station = st.selectbox("Station de retrait", station_names, disabled=not station_names)
    end_station = st.selectbox("Station de retour", station_names, disabled=not station_names)
    duree = st.number_input("Durée (heures)", min_value=1, max_value=240, step=1, value=2)
    email = st.text_input("Email (UNIL/EPFL)", placeholder="prenom.nom@unil.ch")

    submitted = st.form_submit_button("Réserver")

# --- 4. Process reservation ---
if submitted:
    if not model or not email or not start_station or not end_station:
        st.error("Tous les champs sont requis.")
        st.stop()

    pickup_ts = datetime.combine(date_ret, heure_ret)
    pickup_iso = pickup_ts.isoformat(sep=" ")

    # IDs des stations
    start_station_id = stations_dict[start_station]
    end_station_id = stations_dict[end_station]

    # 1) Trouver un vélo disponible
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
        st.error("Aucun vélo disponible pour ce modèle à cet instant.")
        st.stop()

    bike_id = bike_res.data[0]["BIKE_ID"]

    # 2) Insérer la réservation
    supabase.table("reservations").insert({
        "BIKE_ID": bike_id,
        "START_STATION_ID": start_station_id,
        "END_STATION_ID": end_station_id,
        "USER_EMAIL": email,
        "PICKUP_TS": pickup_iso,
        "DURATION_HOURS": int(duree),
        "STATUS": "ACTIVE"
    }).execute()

    # 3) Mettre à jour le vélo comme réservé
    supabase.table("bikes").update({"STATUS": "RESERVED"}).eq("BIKE_ID", bike_id).execute()

    # 4) Messages Streamlit
    st.success(f"Réservation confirmée — Vélo ID {bike_id} ({model}) le {pickup_iso} pour {int(duree)}h.")
    st.info(f"📍 Retrait: {start_station} → Retour: {end_station}")