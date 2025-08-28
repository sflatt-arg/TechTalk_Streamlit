import streamlit as st

st.set_page_config(page_title="Parc vélo — Tableau", page_icon="📊", layout="wide")
st.header("📊 Parc vélo — Disponibles / Réservés / Maintenance")

session = get_active_session()

def load_bikes_df():
    q = """
        SELECT 
          b.BIKE_ID, b.MODEL, b.STATUS, 
          s.NAME AS STATION, s.STATION_ID
        FROM BIKES b
        JOIN STATIONS s ON b.STATION_ID = s.STATION_ID
        ORDER BY s.NAME, b.MODEL, b.BIKE_ID
    """
    return session.sql(q).to_pandas()


if st.button("Refresh now"):
    st.rerun()

df = load_bikes_df()

if df.empty:
    st.info("Pas de données. Initialise les tables.")
    st.stop()

# Filtres
c1, c2, c3 = st.columns(3)
with c1:
    status_opts = df["STATUS"].dropna().unique().tolist()
    status_sel = st.multiselect("État", options=status_opts, default=status_opts)
with c2:
    station_opts = df["STATION"].dropna().unique().tolist()
    station_sel = st.multiselect("Station", options=station_opts, default=station_opts)
with c3:
    model_opts = df["MODEL"].dropna().unique().tolist()
    model_sel = st.multiselect("Modèle", options=model_opts, default=model_opts)

mask = (
    df["STATUS"].isin(status_sel) &
    df["STATION"].isin(station_sel) &
    df["MODEL"].isin(model_sel)
)
view = df.loc[mask].reset_index(drop=True)

st.dataframe(
    view,
    use_container_width=True,
    hide_index=True
)

# Petites métriques
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Vélos total", int(df.shape[0]))
with m2:
    st.metric("Disponibles", int((df["STATUS"] == "AVAILABLE").sum()))
with m3:
    st.metric("Réservés", int((df["STATUS"] == "RESERVED").sum()))
