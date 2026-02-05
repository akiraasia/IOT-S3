import requests
import numpy as np
import datetime
from PIL import Image
import io
import math

# Open-Meteo for Weather Data
def get_real_time_weather(lat, lon):
    """
    Fetch weather conditions specifically for Physics-Based Cloud Removal.
    We need: Temperature (Air Density) and Humidity (Water Vapor density).
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover&timezone=auto"
    try:
        response = requests.get(url).json()
        current = response.get('current', {})
        return {
            "temperature": current.get('temperature_2m', 20),      # Celsius
            "humidity": current.get('relative_humidity_2m', 50),   # Percent
            "pressure": current.get('surface_pressure', 1013),     # hPa
            "cloud_cover": current.get('cloud_cover', 0),          # Percent
            "timestamp": current.get('time', "Unknown")
        }
    except Exception as e:
        print(f"Weather Fetch Error: {e}")
        return {"error": str(e), "temperature": 20, "humidity": 50, "pressure": 1013}

def lat_lon_to_tile(lat, lon, z):
    """
    Converts Lat/Lon to Web Mercator Tile Coordinates (Google/OSM/Bing convention).
    """
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return x, y

def fetch_satellite_imagery(lat, lon, zoom=9):
    """
    Fetches:
    1. REAL 'Cloudy' Image from NASA GIBS (MODIS/VIIRS) for TODAY.
    2. PRIOR 'Clear' Image from ArcGIS World Imagery.
    
    Note: NASA GIBS zoom levels are limited (Max ~9-10 usually for MODIS).
    """
    x, y = lat_lon_to_tile(lat, lon, zoom)
    
    # --- 1. Fetch Real-Time Data (NASA GIBS) ---
    # We use MODIS Terra which has daily coverage.
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # NASA GIBS WMTS URL Pattern
    # Layer: MODIS_Terra_CorrectedReflectance_TrueColor
    # MatrixSet: GoogleMapsCompatible_Level9 (This matches standard Web Mercator tile grid)
    nasa_url = (
        f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"MODIS_Terra_CorrectedReflectance_TrueColor/default/{today}/"
        f"GoogleMapsCompatible_Level9/{zoom}/{y}/{x}.jpg"
    )
    
    try:
        print(f"Fetching NASA GIBS: {nasa_url}")
        resp_nasa = requests.get(nasa_url, timeout=10)
        
        if resp_nasa.status_code == 200:
            cloudy_img = Image.open(io.BytesIO(resp_nasa.content)).convert('RGB')
        else:
            # Fallback for "Today" if data not yet processed (UTC lag), try Yesterday
            print("NASA Data for today not ready, trying yesterday...")
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            nasa_url = nasa_url.replace(today, yesterday)
            resp_nasa = requests.get(nasa_url, timeout=10)
            if resp_nasa.status_code == 200:
                cloudy_img = Image.open(io.BytesIO(resp_nasa.content)).convert('RGB')
            else:
                # Absolute Fallback: Transparent placeholder
                cloudy_img = Image.new('RGB', (256, 256), color=(200, 200, 200))
    except Exception as e:
        print(f"NASA GIBS Fetch Error: {e}")
        cloudy_img = Image.new('RGB', (256, 256), color=(100, 100, 100))

    # --- 2. Fetch Prior (ArcGIS) ---
    # We might need a higher zoom for ArcGIS usually, but for pixel-matching we should keep them same size.
    # However, NASA is stuck at zoom 9 for this layer usually.
    # If we request ArcGIS at zoom 9, it will look blurry but match.
    # Let's request ArcGIS at the same zoom for alignment.
    arcgis_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    
    try:
        resp_prior = requests.get(arcgis_url, timeout=10)
        resp_prior.raise_for_status()
        prior_img = Image.open(io.BytesIO(resp_prior.content)).convert('RGB')
    except Exception as e:
        print(f"ArcGIS Prior Fetch Error: {e}")
        prior_img = Image.new('RGB', (256, 256), color=(0, 100, 0))

    # Resize to ensure match (NASA GIBS tiles are 256x256)
    cloudy_img = cloudy_img.resize((256, 256))
    prior_img = prior_img.resize((256, 256))

    return cloudy_img, prior_img
