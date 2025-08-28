import streamlit as st
import pandas as pd
from datetime import datetime

st.header("🚲 Retour de vélo")

# Récupère la session Snowflake
session = get_active_session()

# Utilitaire simple pour échapper les quotes dans les strings SQL
def sql_escape(s: str) -> str:
    return (s or "").replace("'", "''")

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

if submitted:
    if not bike_id or not email:
        st.error("L'ID du vélo et l'email sont requis.")
        st.stop()
    
    email_q = sql_escape(email)
    
    # 1) Vérifier que la réservation existe et correspond
    reservation_check = session.sql(f"""
        SELECT 
            r.RESERVATION_ID,
            r.BIKE_ID,
            r.USER_EMAIL,
            r.START_STATION_ID,
            r.END_STATION_ID,
            r.PICKUP_TS,
            r.DURATION_HOURS,
            r.STATUS,
            s1.NAME as START_STATION_NAME,
            s2.NAME as END_STATION_NAME,
            b.STATUS as CURRENT_BIKE_STATUS,
            b.STATION_ID
        FROM RESERVATIONS r
        JOIN STATIONS s1 ON r.START_STATION_ID = s1.STATION_ID
        JOIN STATIONS s2 ON r.END_STATION_ID = s2.STATION_ID  
        JOIN BIKES b ON r.BIKE_ID = b.BIKE_ID
        WHERE r.BIKE_ID = {int(bike_id)}
          AND r.USER_EMAIL = '{email_q}'
          AND r.STATUS = 'ACTIVE'
        ORDER BY r.PICKUP_TS DESC
        LIMIT 1
    """).collect()
    
    if not reservation_check:
        st.error("❌ Aucune réservation active trouvée pour ce vélo et cet email.")
        st.info("Vérifiez que :")
        st.write("- L'ID du vélo est correct")
        st.write("- L'email correspond exactement à celui utilisé lors de la réservation")
        st.write("- La réservation n'a pas déjà été terminée")
        st.stop()
    
    # Récupérer les données de la réservation
    reservation = reservation_check[0]
    reservation_id = reservation[0]
    start_station_name = reservation[8]
    end_station_name = reservation[9]
    current_bike_status = reservation[10]
    pickup_ts = reservation[5]
    duration_hours = reservation[6]
    end_station_id = reservation[4]
    
    # Vérifier que le vélo est bien réservé
    if current_bike_status != 'RESERVED':
        st.warning(f"⚠️ Le vélo {bike_id} n'est pas dans l'état 'RESERVED' (état actuel: {current_bike_status})")
        st.info("Le retour va quand même être traité.")
    
    # Afficher les détails de la réservation
    st.success("✅ Réservation trouvée !")
    

    st.write("**Détails de la réservation:**")
    st.write(f"- Vélo ID: {bike_id}")
    st.write(f"- Email: {email}")
    st.write(f"- Retrait: {start_station_name}")
    st.write(f"- Retour prévu: {end_station_name}")
    
     
    # Confirmer le retour
    if st.button("✅ Confirmer le retour", type="primary"):
        try:
            # 2) Mettre à jour le vélo pour qu'il soit disponible à la station de retour
            session.sql(f"""
                UPDATE BIKES 
                SET 
                    STATUS = 'AVAILABLE',
                    STATION_ID = {end_station_id}
                WHERE BIKE_ID = {int(bike_id)}
            """).collect()
            
            # 3) Marquer la réservation comme terminée
            session.sql(f"""
                UPDATE RESERVATIONS 
                SET STATUS = 'FINISHED'
                WHERE RESERVATION_ID = {reservation_id}
            """).collect()
            
            st.success(f"🎉 Vélo {bike_id} retourné avec succès !")
            st.success(f"📍 Le vélo est maintenant disponible à la station: **{end_station_name}**")
            
            # Afficher un résumé
            st.balloons()
            
            with st.expander("Résumé du retour"):
                st.write("**Actions effectuées:**")
                st.write(f"- Vélo {bike_id} marqué comme 'AVAILABLE'")
                st.write(f"- Vélo repositionné à la station: {end_station_name}")
                st.write(f"- Réservation {reservation_id} marquée comme 'FINISHED'")
                st.write(f"- Heure de retour: {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}")
            
        except Exception as e:
            st.error(f"❌ Erreur lors du retour du vélo: {str(e)}")
            st.info("Veuillez contacter l'administrateur si le problème persiste.")

# Section d'aide
with st.expander("ℹ️ Aide"):
    st.write("""
    **Comment retourner un vélo :**
    
    1. Saisissez l'ID du vélo (numéro affiché sur le vélo)
    2. Entrez l'email utilisé lors de la réservation
    3. Cliquez sur "Retourner le vélo"
    4. Vérifiez les informations et confirmez le retour
    
    **En cas de problème :**
    - Vérifiez que l'ID du vélo est correct
    - Assurez-vous d'utiliser exactement le même email que lors de la réservation
    - Contactez le support si le vélo n'apparaît pas comme réservé
    """)
