import streamlit as st
import pandas as pd
import numpy as np
from scraper import fetch_satellite_imagery, get_real_time_weather
from inference import run_physics_inference
from PIL import Image

st.set_page_config(page_title="IOT-S3: Physics-Based Cloud Removal", layout="wide")

st.title("🛰️ IOT-S3: Real-World Physics Cloud Removal")
st.markdown("### Using Thermodynamic Atmospheric Modelling & NASA GIBS Data")
st.markdown("---")

# Sidebar for location input
st.sidebar.header("Target Coordinates")
# Default: Someplace likely to be cloudy or interesting
lat = st.sidebar.number_input("Latitude", value=51.5074, format="%.4f") # London
lon = st.sidebar.number_input("Longitude", value=-0.1278, format="%.4f")
zoom = st.sidebar.slider("Zoom Level", 5, 12, 9) # NASA GIBS limits zoom

st.sidebar.info("Note: NASA GIBS 'TrueColor' is available up to Zoom ~9. Higher zooms will be upscaled.")

if st.sidebar.button("Initialize Satellite Link"):
    with st.spinner("Acquiring Signal... Synchronizing with NASA Terra/Aqua..."):
        # 1. Get Real-Time Physics Data
        weather = get_real_time_weather(lat, lon)
        
        # 2. Fetch Imagery (Real NASA Cloud + ArcGIS Prior)
        cloudy, prior = fetch_satellite_imagery(lat, lon, zoom)
        
        if cloudy is not None and prior is not None:
            st.success("Signal Acquired. Data Stream Active.")
            
            # Display Physics Parameters
            st.markdown("#### 🌡️ Atmospheric Physics Parameters")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Temperature", f"{weather.get('temperature')} °C", help="Affects Air Density")
            col2.metric("Humidity", f"{weather.get('humidity')} %", help="Determines Water Vapor Density (Scattering)")
            col3.metric("Pressure", f"{weather.get('pressure')} hPa")
            col4.metric("Cloud Cover", f"{weather.get('cloud_cover')} %")
            
            # 3. Perform Physics Inference
            st.markdown("### 🧬 Radiative Transfer Restoration")
            processed, transmission_map = run_physics_inference(cloudy, prior, weather)
            
            row1 = st.columns(4)
            with row1[0]:
                st.image(cloudy, caption="Step 1: NASA GIBS Real-Time (Today)", use_container_width=True)
            with row1[1]:
                st.image(transmission_map, caption="Step 2: Calculated Transmission Map (Opacity)", use_container_width=True)
            with row1[2]:
                st.image(prior, caption="Step 3: Temporal Prior (GIS Reference)", use_container_width=True)
            with row1[3]:
                st.image(processed, caption="Result: Physics-Restored Output", use_container_width=True)
            
            st.info(f"""
            **How it works:**
            1. **Inversion**: We used the Humidity ({weather.get('humidity')}%) to calculate the Scattering Coefficient.
            2. **Transmission**: The Transmission Map shows where light was blocked (Darker = More Blocked).
            3. **Fusion**: We boosted valid signals and filled opaque gaps with the Prior using the GPS fix.
            """)
        else:
            st.error("Signal Lost. Could not fetch imagery.")

else:
    st.info("Enter coordinates and click 'Initialize Satellite Link' to connect.")

st.markdown("---")
st.caption("Powered by NASA GIBS, Open-Meteo, & Beer-Lambert Law Physics.")
