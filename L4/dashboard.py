import streamlit as st
import yaml

with open("../config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

st.set_page_config(layout="wide", page_title="Security Monitor")
st.title("VLM Surveillance Dashboard")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Video Feed")
    st.info("Video feed will be displayed here...")

with col2:
    st.subheader("System Status")
    st.metric(label="Persons Detected", value=0)
    st.metric(label="Vehicles Detected", value=0)
    st.metric(label="Alerts", value=0)
    
    st.subheader("Alerts Log")
    st.warning("No suspicious activity detected yet.")
