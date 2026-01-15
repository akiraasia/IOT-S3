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

import math

# ArcGIS World Imagery Tile Service
def fetch_satellite_imagery(lat, lon, zoom=15):
    """
    Fetches real-time satellite imagery using ArcGIS World Imagery Tile Service.
    Calculates the specific tile (x, y) for a given Lat/Lon and Zoom.
    """
    def lat_lon_to_tile(lat, lon, z):
        lat_rad = math.radians(lat)
        n = 2.0 ** z
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
        return x, y

    x, y = lat_lon_to_tile(lat, lon, zoom)
    
    # ArcGIS Tile URL
    # We use zoom, x, y coordinates
    tile_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    
    # To simulate a 'prior' (cloud-free), we can fetch the same tile but slightly offset or just use a placeholder
    # In a real app, 'prior' would be cached historical data.
    
    try:
        resp = requests.get(tile_url, timeout=10)
        resp.raise_for_status()
        
        main_img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        
        # For demonstration of 'cloud removal', we artificially add realistic clouds
        prior = main_img.copy()
        
        # Create a cloudy version with realistic soft clouds
        cloudy_np = np.array(main_img).astype(np.float32)
        h, w, c = cloudy_np.shape
        
        # Soft cloud generation
        for _ in range(np.random.randint(4, 8)):
            r = np.random.randint(30, 80)
            cx, cy = np.random.randint(0, w, 2)
            
            y_grid, x_grid = np.ogrid[:h, :w]
            dist_sq = (x_grid - cx)**2 + (y_grid - cy)**2
            
            # Gaussian soft cloud mask
            mask = np.exp(-dist_sq / (2 * (r/2)**2))
            mask = np.expand_dims(mask, axis=-1)
            
            # White cloud color with some variation
            cloud_color = np.array([245, 245, 255])
            cloudy_np = cloudy_np * (1 - mask*0.8) + cloud_color * (mask*0.8)
            
        cloudy = Image.fromarray(cloudy_np.astype(np.uint8))
        return cloudy, prior
    except Exception as e:
        print(f"Imagery Fetch Error (ArcGIS): {e}")
        return None, None
