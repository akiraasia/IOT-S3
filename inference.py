import numpy as np
import PIL.Image
import scipy.ndimage as ndimage

def run_physics_inference(cloudy_img, prior_img, weather_data):
    """
    Physics-Based Cloud Removal using Atmospheric Light Intensity Adjustment.
    
    Theory:
    I_res = (I_obs - I_path) / Transmission
    
    Where:
    - Transmission (t) is estimated from Humidity (Water Vapor Density).
    - I_path (Airlight) is estimated from the brightest pixels (clouds).
    - If t < Threshold, we fallback to GPS-Prior.
    """
    # 1. Prepare Data
    cloudy = np.array(cloudy_img).astype(np.float32) / 255.0
    prior = np.array(prior_img).astype(np.float32) / 255.0
    
    temp = weather_data.get('temperature', 20)
    humidity = weather_data.get('humidity', 50)
    
    # 2. Physics Model Parameters
    # beta_scalt: Scattering coefficient. Higher humidity = more scattering/haze.
    # Simple linear approximation for demo:
    # 0.0 at 0% humidity, 2.0 at 100% humidity.
    beta_scat = (humidity / 100.0) * 2.5 + 0.1
    
    # Transmission Map (Global estimation for this tile based on physics)
    # T(x) = e^(-beta * d(x))
    # We assume a constant 'cloud depth' (d) for a single tile for simplicity,
    # or we can derive d(x) from the 'Dark Channel Prior'.
    
    # Let's estimate local 'Optical Depth' using the Dark Channel Prior method
    # Dark Channel = min(RGB)
    dark_channel = np.min(cloudy, axis=2)
    
    # Atmospheric Light (A): Estimate from the top 0.1% brightest pixels in dark channel
    num_pixels = dark_channel.size
    num_brightest = int(max(num_pixels * 0.001, 1))
    indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
    flat_cloudy = cloudy.reshape(-1, 3)
    A = np.mean(flat_cloudy[indices], axis=0) # [R, G, B] of the "Cloud"
    
    # avoid division by zero
    A = np.maximum(A, 0.1) 
    
    # Transmission (t)
    # Omega = 0.95 (amount of haze to keep for realism, but we want to remove it, so 1.0)
    # We modulate omega with Humidity. High humidity = we need to remove more.
    omega = 0.5 + (humidity / 200.0) # 0.5 to 1.0
    
    transmission = 1 - omega * np.min(cloudy / A, axis=2)
    transmission = np.maximum(transmission, 0.1) # Threshold to avoid noise
    
    # Refine Transmission (Guided Filter substitute -> Gaussian for speed)
    transmission = ndimage.gaussian_filter(transmission, sigma=2)
    
    # 3. Radiance Recovery (Inverting the Atmosphere)
    # J = (I - A) / t + A
    output = np.zeros_like(cloudy)
    for c in range(3):
        output[:, :, c] = (cloudy[:, :, c] - A[c]) / transmission + A[c]
        
    output = np.clip(output, 0, 1)
    
    # 4. GPS-Prior Fusion
    # Where transmission is very low (Thick Clouds), the Physics model amplifies noise.
    # We define a "Confidence" mask based on transmission.
    # If Transmission < 0.3 -> It's a thick cloud -> Use Prior.
    # If Transmission > 0.7 -> It's clear/hazy -> Use Restored Physics Output.
    
    confidence = np.expand_dims(transmission, axis=-1)
    
    # Soft blending
    # Blend = Confidence * Restored + (1 - Confidence) * Prior
    # However, we want to allow the "Restored" image to dominate where it's good,
    # and "Prior" to take over ONLY where it's bad.
    
    # Enhance confidence contrast to make a sharper decision
    weight = np.clip((confidence - 0.2) * 5, 0, 1) # Sigmoid-like transition
    
    final_output = output * weight + prior * (1 - weight)
    
    final_output = output * weight + prior * (1 - weight)
    
    # 5. Visual Diagnostics
    # Return both the result and the Transmission Map (visualized as heatmap)
    trans_map = (transmission * 255).astype(np.uint8)
    if len(trans_map.shape) == 2:
        trans_map = np.stack([trans_map]*3, axis=-1) # Grayscale to RGB
        
    return PIL.Image.fromarray((final_output * 255).astype(np.uint8)), PIL.Image.fromarray(trans_map)

