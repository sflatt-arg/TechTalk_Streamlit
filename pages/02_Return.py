import streamlit as st
import pandas as pd
from datetime import datetime
from supabase import create_client

# Load credentials from Streamlit secrets
url = st.secrets["supabase"]["url"]
key = st.secrets["supabase"]["key"]

supabase = create_client(url, key)


st.header("🚲 Retour de vélo")

# Initialiser session_state
if 'reservation_found' not in st.session_state:
    st.session_state.reservation_found = False
if 'reservation_data' not in st.session_state:
    st.session_state.reservation_data = None

# Formulaire de recherche
with st.form("return_form"):
    st.subheader("Informations de retour")
    
    bike_id = st.number_input(
        "ID du vélo", 
        min_value=1, 
        step=1, 
        help="L'ID du vélo que vous souhaitez retourner"
    )
    
    email = st.text_input(
        "Email de réservation", 
        placeholder="prenom.nom@unil.ch",
        help="L'email utilisé lors de la réservation"
    )
    
    submitted = st.form_submit_button("Retourner le vélo")

# Traitement de la recherche
if submitted:
    if not bike_id or not email:
        st.error("L'ID du vélo et l'email sont requis.")
        st.session_state.reservation_found = False
    else:
        # Votre requête Supabase...
        r = (
            supabase.table("reservations")
            .select(
                "RESERVATION_ID,BIKE_ID,USER_EMAIL,START_STATION_ID,END_STATION_ID,PICKUP_TS,DURATION_HOURS,STATUS,"
                "bikes(STATUS,STATION_ID),"
                "start_station:stations!reservations_START_STATION_ID_fkey(NAME),"
                "end_station:stations!reservations_END_STATION_ID_fkey(NAME)"
            )
            .eq("BIKE_ID", int(bike_id))
            .eq("USER_EMAIL", email)
            .eq("STATUS", "ACTIVE")
            .order("PICKUP_TS", desc=True)
            .limit(1)
            .execute()
        )
        
        if r.data:
            row = r.data[0]
            reservation_check = {
                **{k: row[k] for k in ["RESERVATION_ID","BIKE_ID","USER_EMAIL","START_STATION_ID","END_STATION_ID","PICKUP_TS","DURATION_HOURS","STATUS"]},
                "current_bike_status": row["bikes"]["STATUS"],
                "bike_station_id": row["bikes"]["STATION_ID"],
                "start_station_name": row["start_station"]["NAME"],
                "end_station_name": row["end_station"]["NAME"],
            }
            
            # Sauvegarder dans session_state
            st.session_state.reservation_found = True
            st.session_state.reservation_data = reservation_check
        else:
            st.error("Aucune réservation active trouvée.")
            st.session_state.reservation_found = False

# Affichage de la réservation (persiste après le clic du bouton)
if st.session_state.reservation_found and st.session_state.reservation_data:
    reservation_check = st.session_state.reservation_data
    
    # Extraire les variables
    reservation_id = reservation_check["RESERVATION_ID"]
    start_station_name = reservation_check["start_station_name"]
    end_station_name = reservation_check["end_station_name"]
    current_bike_status = reservation_check["current_bike_status"]
    pickup_ts = reservation_check["PICKUP_TS"]
    duration_hours = reservation_check["DURATION_HOURS"]
    end_station_id = reservation_check["END_STATION_ID"]
    bike_id = reservation_check["BIKE_ID"]
    email = reservation_check["USER_EMAIL"]
    
    # Affichage des infos
    st.success("Réservation trouvée !")
    
    st.subheader("Détails de la réservation:")
    st.write(f"• **Vélo ID:** {bike_id}")
    st.write(f"• **Email:** {email}")
    st.write(f"• **Retrait:** {start_station_name}")
    st.write(f"• **Retour prévu:** {end_station_name}")
    
    # LE BOUTON - maintenant il persiste !
    if st.button("✅ Confirmer le retour", type="primary"):
        
        try:
            # 1) Mise à jour du vélo - SANS select()
            bike_update = (
                supabase.table("bikes")
                .update({"STATUS": "AVAILABLE", "STATION_ID": int(end_station_id)})
                .eq("BIKE_ID", int(bike_id))
                .execute()
            )
            
            # 2) Mise à jour de la réservation 
            st.write("Mise à jour de la réservation")
            res_update = (
                supabase.table("reservations")
                .update({"STATUS": "FINISHED"})
                .eq("RESERVATION_ID", int(reservation_id))
                .eq("STATUS", "ACTIVE")
                .execute()
            )
            
            # 3) Vérification et feedback
            if bike_update.data and res_update.data:
                st.success(f"Vélo {bike_id} retourné avec succès !")
                st.success(f"Le vélo est maintenant disponible à la station: **{end_station_name}**")
                st.balloons()
                
                # Reset session state après succès
                st.session_state.reservation_found = False
                st.session_state.reservation_data = None
                
            else:
                st.error("❌ Une des mises à jour a échoué")
                if not bike_update.data:
                    st.error("- Échec mise à jour vélo")
                if not res_update.data:
                    st.error("- Échec mise à jour réservation")
                
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            st.exception(e)