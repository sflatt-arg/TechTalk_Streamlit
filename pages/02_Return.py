import streamlit as st
import pandas as pd
from datetime import datetime
from supabase_client import supabase

st.header("Bike return")

# 1. Initialise session_state
# To persist data after button click
if 'reservation_found' not in st.session_state:
    st.session_state.reservation_found = False
if 'reservation_data' not in st.session_state:
    st.session_state.reservation_data = None

# 2. Formulaire de recherche
with st.form("return_form"):
    st.subheader("Return a bike")
    st.write("To return your bike, enter the bike ID and the email used during the reservation.")
    
    bike_id = st.number_input(
        "Bike ID", 
        min_value=1, 
        step=1, 
        help="The bike ID is written on the bike"
    )
    
    email = st.text_input(
        "Booking email", 
        placeholder="firstname.lastname@epfl.ch",
        help="Email used during the reservation"
    )
    
    submitted = st.form_submit_button("Retourner le vélo")

# Retrieves the reservation from Supabase
if submitted:
    if not bike_id or not email:
        st.error("Bike ID and email are required.")
        st.session_state.reservation_found = False
    else:
        # Supabase query to find the latest active reservation for this bike and email
        r = (
            supabase.table("reservations")
            .select(
                "RESERVATION_ID,BIKE_ID,USER_EMAIL,START_STATION_ID,END_STATION_ID,PICKUP_TS,DURATION_HOURS,STATUS,"
                "bikes(STATUS,STATION_ID),"
                "START_STATION:stations!reservations_START_STATION_ID_fkey(NAME),"
                "END_STATION:stations!reservations_END_STATION_ID_fkey(NAME)"
            )
            .eq("BIKE_ID", int(bike_id))
            .eq("USER_EMAIL", email)
            .eq("STATUS", "ACTIVE")
            .order("PICKUP_TS", desc=True)
            .limit(1)
            .execute()
        )
        
        if r.data:
            # If a reservation is found, extract and flatten the data
            row = r.data[0]
            reservation_check = {
                **{k: row[k] for k in ["RESERVATION_ID","BIKE_ID","USER_EMAIL","START_STATION_ID","END_STATION_ID","PICKUP_TS","DURATION_HOURS","STATUS"]},
                "current_bike_status": row["bikes"]["STATUS"],
                "bike_station_id": row["bikes"]["STATION_ID"],
                "start_station_name": row["START_STATION"]["NAME"],
                "end_station_name": row["END_STATION"]["NAME"],
            }
            
            # Save in session state (to use the data later)
            st.session_state.reservation_found = True
            st.session_state.reservation_data = reservation_check
        else:
            st.error("Aucune réservation active trouvée.")
            st.session_state.reservation_found = False

# Affichage de la réservation (persiste après le clic du bouton)
if st.session_state.reservation_found and st.session_state.reservation_data:
    reservation_check = st.session_state.reservation_data
    
    # Extract data from the saved reservation
    reservation_id = reservation_check["RESERVATION_ID"]
    start_station_name = reservation_check["start_station_name"]
    end_station_name = reservation_check["end_station_name"]
    current_bike_status = reservation_check["current_bike_status"]
    pickup_ts = reservation_check["PICKUP_TS"]
    duration_hours = reservation_check["DURATION_HOURS"]
    end_station_id = reservation_check["END_STATION_ID"]
    bike_id = reservation_check["BIKE_ID"]
    email = reservation_check["USER_EMAIL"]
    
    # Print reservation details
    st.success("Reservation found !")
    
    st.subheader("Reservation details:")
    st.write(f"• **Bike ID:** {bike_id}")
    st.write(f"• **Email:** {email}")
    st.write(f"• **Pickup station:** {start_station_name}")
    st.write(f"• **Return station:** {end_station_name}")
    
    if st.button("Confirm return", type="primary"):
        # If confirm button is clicked, process the return
        try:
            # 1) Update bike tables : set STATUS to AVAILABLE and update STATION_ID
            bike_update = (
                supabase.table("bikes")
                .update({"STATUS": "AVAILABLE", "STATION_ID": int(end_station_id)})
                .eq("BIKE_ID", int(bike_id))
                .execute()
            )
            
            # 2) Update reservation table : set STATUS to FINISHED 
            res_update = (
                supabase.table("reservations")
                .update({"STATUS": "FINISHED"})
                .eq("RESERVATION_ID", int(reservation_id))
                .eq("STATUS", "ACTIVE")
                .execute()
            )
            
            # 3) Check and feedback
            if bike_update.data and res_update.data:
                st.success(f"Bike {bike_id} successfully returned !")
                st.success(f"Bike is now available in station **{end_station_name}**")
                
                # Reset session state après succès
                st.session_state.reservation_found = False
                st.session_state.reservation_data = None
                
            else:
                st.error("Data update failed:")
                if not bike_update.data:
                    st.error("- Bike update failed")
                if not res_update.data:
                    st.error("- Reservation update failed")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)} - contact the app administrator.")
            st.exception(e)