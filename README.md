# 🛰️ IOT-S3: Intelligent Cloud Shadow Removal

## Overview
IOT-S3 is a real-time satellite imagery reconstruction application. It uses a **Temporal-UNet** with a **"Prior Guessing"** algorithm to remove cloud shadows from satellite imagery by leverages multi-source data integration (GPS-style fix).

## Features
- **Real-time Scraper**: Fetches latest data from Open-Meteo and simulated satellite feeds.
- **Prior Guessing**: Uses previous clear-sky metadata to "guess" the terrain beneath clouds.
- **Interactive UI**: Built with Streamlit for easy coordinate selection.

## Running in GitHub Codespaces
1. Open this repository in a Codespace.
2. The environment will auto-install dependencies from `requirements.txt`.
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. A preview window will automatically open (Port 8501).

## Literature Review
Based on 23 research papers (2020-2025) including:
- **Chaoui et al., 2024**: Focus on ground-based solar imagery.
- **Li et al., 2023**: Multimodal Remote Sensing Fusion.
- **Wang et al., 2025**: SAR-Fused Thick Cloud Removal.

---
Developed for the IOT-S3 Project.
