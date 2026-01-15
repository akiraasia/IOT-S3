import requests
import numpy as np
import datetime
from PIL import Image
import io

# Open-Meteo for Weather Data
def get_real_time_weather(lat, lon):
    """Fetch cloud cover and weather conditions."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=cloud_cover,relative_humidity,surface_pressure&timezone=auto"
    try:
        response = requests.get(url).json()
        current = response.get('current', {})
        return {
            "cloud_cover": current.get('cloud_cover', 0),
            "humidity": current.get('relative_humidity', 0),
            "pressure": current.get('surface_pressure', 0),
            "timestamp": current.get('time', "Unknown")
        }
    except Exception as e:
        return {"error": str(e)}

# Mock Imagery Scraper (Simulating Google Earth Engine / Public API)
def fetch_satellite_imagery(lat, lon, zoom=15):
    """
    Simulates fetching a real-time satellite snapshot.
    In a full production environment, this would use GEE API authentication.
    """
    # Using a placeholder implementation that returns a real-world snapshot for demo
    # In Codespaces, we use pre-fetched/available tiles for reliability
    sample_img_url = "https://raw.githubusercontent.com/BUPTLdy/RICE_DATASET/master/RICE-I/cloudy/10.png"
    prior_img_url = "https://raw.githubusercontent.com/BUPTLdy/RICE_DATASET/master/RICE-I/ground_truth/10.png"
    
    def get_img(url):
        resp = requests.get(url)
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        return img.resize((256, 256))

    try:
        cloudy = get_img(sample_img_url)
        prior = get_img(prior_img_url)
        return cloudy, prior
    except Exception as e:
        print(f"Imagery Fetch Error: {e}")
        return None, None
