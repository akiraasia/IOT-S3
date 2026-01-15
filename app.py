import streamlit as st
import pandas as pd
import numpy as np
from scraper import fetch_satellite_imagery, get_real_time_weather
from inference import run_inference
from PIL import Image

st.set_page_config(page_title="IOT-S3: Real-Time Cloud Shadow Removal", layout="wide")

st.title("🛰️ IOT-S3: Intelligent Cloud Shadow Removal")
st.markdown("---")

# Sidebar for location input
st.sidebar.header("Location Settings")
lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.4f")
zoom = st.sidebar.slider("Zoom Level", 10, 20, 15)

if st.sidebar.button("Fetch & Process Imagery"):
    with st.spinner("Fetching real-time weather and satellite data..."):
        # 1. Get Weather Data
        weather = get_real_time_weather(lat, lon)
        
        # 2. Fetch Imagery (Cloudy and Prior)
        cloudy, prior = fetch_satellite_imagery(lat, lon, zoom)
        
        if cloudy and prior:
            st.success("Data Fetched Successfully!")
            
            # Display Weather Info
            col1, col2, col3 = st.columns(3)
            col1.metric("Cloud Cover", f"{weather['cloud_cover']}%")
            col2.metric("Humidity", f"{weather['humidity']}%")
            col3.metric("Pressure", f"{weather['pressure']} hPa")
            
            # 3. Perform Inference
            st.markdown("### Reconstruction Pipeline")
            processed = run_inference(cloudy, prior)
            
            row1 = st.columns(3)
            row1[0].image(cloudy, caption="Real-time Scraped Imagery (Cloudy)", use_container_width=True)
            row1[1].image(prior, caption="Temporal Prior (Last Clear Image)", use_container_width=True)
            row1[2].image(processed, caption="Cloud-Free Output", use_container_width=True)
            
            st.info("The output demonstrates the 'Prior Guessing' algorithm converging on the true terrain based on the 4+2 channel input fix.")
        else:
            st.error("Could not fetch imagery. Please check API settings.")

else:
    st.info("Enter coordinates and click 'Fetch & Process Imagery' to start.")

st.markdown("---")
st.caption("Powered by Temporal-UNet and Open-Meteo APIs. Developed for IOT-S3 Project.")
