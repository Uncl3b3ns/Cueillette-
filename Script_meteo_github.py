import os
import datetime
import numpy as np
import requests
import time
from tqdm import tqdm

# Paramètres
DAYS_BEFORE = 6
DAYS_AFTER = 7
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
LAT_MIN, LAT_MAX = 41.0, 51.0
LON_MIN, LON_MAX = -5.0, 10.0
LAT_STEP, LON_STEP = 4.0, 4.0  # Grille moins dense
MAX_RETRIES = 3
API_CALL_DELAY = 0.2  # Pause entre les appels
TIMEOUT = 30  # Timeout augmenté

def create_france_grid():
    """Crée une grille de points pour la France."""
    lats = np.arange(LAT_MIN, LAT_MAX + LAT_STEP, LAT_STEP)
    lons = np.arange(LON_MIN, LON_MAX + LON_STEP, LON_STEP)
    return [(lat, lon) for lat in lats for lon in lons]

def fetch_weather(lat, lon, day, retries=MAX_RETRIES):
    """Récupère les données météo pour un point spécifique avec retry."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": DAILY_VARS,
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "timezone": "auto"
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT)
            response.raise_for_status()  # Vérifie les erreurs HTTP
            return response.json().get("daily", {})
        except requests.exceptions.RequestException as e:
            print(f"[Retry {attempt + 1}/{retries}] Échec pour lat={lat}, lon={lon}: {e}")
            time.sleep(API_CALL_DELAY * 2)  # Pause entre les retries
    return None

def download_weather_data(day):
    """Télécharge les données météo pour un jour donné."""
    points = create_france_grid()
    data = {}
    for lat, lon in tqdm(points, desc=f"Téléchargement météo pour {day}", ncols=80):
        daily_data = fetch_weather(lat, lon, day)
        if daily_data:
            data[(lat, lon)] = daily_data
    return data

if __name__ == "__main__":
    today = datetime.date.today()
    for offset in range(-DAYS_BEFORE, DAYS_AFTER + 1):
        day = today + datetime.timedelta(days=offset)
        print(f"\n[Traitement] Téléchargement des données pour {day}...")
        weather_data = download_weather_data(day)
        print(f"[Terminé] Téléchargement des données pour le {day}.")

