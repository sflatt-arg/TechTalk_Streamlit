import streamlit as st
from datetime import datetime
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Réservation de vélo", page_icon="📝", layout="centered")
st.header("📝 Réservation d’un vélo")

# Récupère la session Snowflake (fourni automatiquement par Streamlit in Snowflake)
session = get_active_session()

# Utilitaire simple pour échapper les quotes dans les strings SQL
def sql_escape(s: str) -> str:
    return (s or "").replace("'", "''")

# Modèles disponibles (distincts), même si certains vélos sont en maintenance
models_rows = session.sql("SELECT DISTINCT MODEL FROM BIKES ORDER BY MODEL").collect()
models = [r[0] for r in models_rows] if models_rows else []

# Stations disponibles (nom et ID)
stations_rows = session.sql("SELECT STATION_ID, NAME FROM STATIONS ORDER BY NAME").collect()
stations_dict = {r[1]: r[0] for r in stations_rows}  # {name: station_id}
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
    bike_row = session.sql(f"""
        SELECT BIKE_ID
        FROM BIKES
        WHERE MODEL = '{model_q}' 
          AND STATUS = 'AVAILABLE' 
          AND STATION_ID = {start_station_id}
        LIMIT 1
    """).collect()

    if not bike_row:
        st.error("Aucun vélo disponible pour ce modèle à cet instant.")
        st.stop()

    bike_id = bike_row[0][0]

    # 2) Insérer la réservation avec les stations
    session.sql(f"""
        INSERT INTO RESERVATIONS (BIKE_ID, START_STATION_ID, END_STATION_ID, USER_EMAIL, PICKUP_TS, DURATION_HOURS)
        VALUES ({bike_id}, {start_station_id}, {end_station_id}, '{email_q}', '{pickup_iso}', {int(duree)})
    """).collect()
    
    # 3) Marquer le vélo comme réservé
    session.sql(f"UPDATE BIKES SET STATUS = 'RESERVED' WHERE BIKE_ID = {bike_id}").collect()
    
    st.success(f"Réservation confirmée — Vélo ID {bike_id} ({model}) le {pickup_iso} pour {int(duree)}h.")
    st.info(f"📍 Retrait: {start_station} → Retour: {end_station}")