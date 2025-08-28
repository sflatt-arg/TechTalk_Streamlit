import streamlit as st
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Réservation de vélo", page_icon="📝", layout="centered")
st.header("📝 Réservation d’un vélo")



# Utilitaire simple pour échapper les quotes dans les strings SQL
def sql_escape(s: str) -> str:
    return (s or "").replace("'", "''")


# Modèles disponibles (distincts), même si certains vélos sont en maintenance
bikes = pd.read_csv("tables/bikes.csv")
models = sorted(bikes["MODEL"].dropna().unique().tolist())

# Stations disponibles (nom et ID)
stations = pd.read_csv("tables/stations.csv")

# Build {name: station_id} dict, ordered by NAME
stations_sorted = stations.sort_values("NAME")
stations_dict = dict(zip(stations_sorted["NAME"], stations_sorted["STATION_ID"]))
station_names = list(stations_dict.keys())

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

if submitted:
    if not model or not email or not start_station or not end_station:
        st.error("Tous les champs sont requis.")
        st.stop()

    pickup_ts = datetime.combine(date_ret, heure_ret)
    email_q = sql_escape(email)
    model_q = sql_escape(model)
    pickup_iso = pickup_ts.isoformat(sep=" ")

    # Récupérer les IDs des stations
    start_station_id = stations_dict[start_station]
    end_station_id = stations_dict[end_station]
    
    # 1) Trouver un vélo disponible du modèle choisi à la station de retrait
    bike_row = (
    bikes[
        (bikes["MODEL"] == model_q)
        & (bikes["STATUS"] == "AVAILABLE")
        & (bikes["STATION_ID"] == start_station_id)
    ]
    .head(1)  # LIMIT 1
    )

    if not bike_row.empty:
        bike_id = bike_row.iloc[0]["BIKE_ID"]
    else:
        bike_id = None

        if not bike_row:
            st.error("Aucun vélo disponible pour ce modèle à cet instant.")
            st.stop()

        bike_id = bike_row[0][0]


    # --- 2) Insert reservation into reservations.csv ---
    reservations = pd.read_csv("tables/reservations.csv")

    new_row = {
        "BIKE_ID": bike_id,
        "START_STATION_ID": start_station_id,
        "END_STATION_ID": end_station_id,
        "USER_EMAIL": email_q,
        "PICKUP_TS": pickup_iso,
        "DURATION_HOURS": int(duree),
    }

    # Append new row
    reservations = pd.concat([reservations, pd.DataFrame([new_row])], ignore_index=True)

    # Save back to CSV
    reservations.to_csv("tables/reservations.csv", index=False)

    # --- 2) Update bikes.csv to mark the bike as RESERVED ---
    bikes.loc[bikes["BIKE_ID"] == bike_id, "STATUS"] = "RESERVED"
    bikes.to_csv("tables/bikes.csv", index=False)

    # --- 3) Streamlit messages ---
    st.success(f"Réservation confirmée — Vélo ID {bike_id} ({model}) le {pickup_iso} pour {int(duree)}h.")
    st.info(f"📍 Retrait: {start_station} → Retour: {end_station}")