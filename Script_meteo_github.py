import os
import datetime
import numpy as np
import requests
import rasterio
from tqdm import tqdm
from scipy.spatial import cKDTree

# Paramètres
DAYS_BEFORE = 6
DAYS_AFTER = 7
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
LAT_MIN, LAT_MAX = 41.0, 51.0
LON_MIN, LON_MAX = -5.0, 10.0
LAT_STEP, LON_STEP = 2.0, 2.0
TEMP_MIN, TEMP_MAX = 5.0, 35.0
PRECIP_MIN = 5.0
WIND_MAX = 25.0

def create_france_grid():
    """Crée une grille de points pour la France."""
    lats = np.arange(LAT_MIN, LAT_MAX + LAT_STEP, LAT_STEP)
    lons = np.arange(LON_MIN, LON_MAX + LON_STEP, LON_STEP)
    return [(lat, lon) for lat in lats for lon in lons]

def download_weather_data(day):
    """Télécharge les données météo pour un jour donné."""
    points = create_france_grid()
    data = {}
    for lat, lon in tqdm(points, desc=f"Téléchargement météo pour {day}"):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": DAILY_VARS,
            "start_date": day.isoformat(),
            "end_date": day.isoformat(),
            "timezone": "auto"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data[(lat, lon)] = response.json().get("daily", {})
    return data

def interpolate_weather(data, raster_shape, bounds):
    """Interpole les données météo sur la grille raster."""
    lats, lons, values = [], [], []
    for (lat, lon), daily_data in data.items():
        temp_max = daily_data.get("temperature_2m_max", [None])[0]
        if temp_max is not None:
            lats.append(lat)
            lons.append(lon)
            values.append(temp_max)

    if not lats:
        print("Aucune donnée météo valide pour interpolation.")
        return np.zeros(raster_shape)

    tree = cKDTree(np.c_[lats, lons])
    grid_lats, grid_lons = np.linspace(bounds[1], bounds[3], raster_shape[0]), np.linspace(bounds[0], bounds[2], raster_shape[1])
    grid_x, grid_y = np.meshgrid(grid_lons, grid_lats)
    grid_points = np.c_[grid_y.ravel(), grid_x.ravel()]

    _, indices = tree.query(grid_points)
    interpolated = np.array(values)[indices].reshape(raster_shape)
    return interpolated

if __name__ == "__main__":
    today = datetime.date.today()
    for offset in range(-DAYS_BEFORE, DAYS_AFTER + 1):
        day = today + datetime.timedelta(days=offset)
        weather_data = download_weather_data(day)
        print(f"Traitement des données pour le {day} terminé.")
