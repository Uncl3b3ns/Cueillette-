#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
import numpy as np
import requests
import rasterio
from rasterio.warp import transform_bounds
import folium
import base64
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
from rasterio.crs import CRS
from tqdm import tqdm
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# 1) Logique "double requête" pour gros fichiers Google Drive
# ---------------------------------------------------------------------

def _get_confirm_token(response):
    """
    Parcourt les cookies à la recherche d'un paramètre download_warning_xxxx
    qui renferme le token de confirmation pour GDrive.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def _save_response_content(response, destination):
    """
    Écrit le contenu de la réponse HTTP (binaire) vers le fichier local,
    en utilisant des chunks pour ne pas surcharger la mémoire.
    """
    CHUNK_SIZE = 32768  # 32 Ko
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # éviter les chunks vides
                f.write(chunk)

def download_large_file_from_google_drive(file_id, destination):
    """
    Télécharge un fichier volumineux depuis Google Drive en gérant la page
    de confirmation d'avertissement.

    :param file_id: l'ID (valeur après 'id=' dans l'URL GDrive)
    :param destination: chemin local où enregistrer le fichier
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    # 1) Première requête
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    # 2) S'il y a un token, on refait la requête avec confirm=token
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # 3) Écriture binaire du contenu
    _save_response_content(response, destination)

# ---------------------------------------------------------------------
# 2) Notre fonction download_from_gdrive fait appel à la logique ci-dessus
# ---------------------------------------------------------------------

def download_from_gdrive(file_id, local_path):
    """
    Vérifie si le fichier local existe déjà, sinon télécharge via la
    logique "gros fichier Google Drive".
    """
    if os.path.exists(local_path):
        print(f"[SKIP] {local_path} existe déjà. Suppression si vous souhaitez re-télécharger.")
        return

    print(f"[DOWNLOAD] Google Drive ID={file_id} => {local_path}")
    try:
        download_large_file_from_google_drive(file_id, local_path)
        print(f"[OK] Téléchargé : {local_path}")
    except Exception as e:
        print(f"[ERROR] Échec du téléchargement GDrive ID={file_id} : {e}")

# ---------------------------------------------------------------------
# 3) Chargement des variables d'environnement (GitHub Token, etc.)
# ---------------------------------------------------------------------

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER   = "Uncl3b3ns"
REPO_NAME    = "Cueillette-"

# ---------------------------------------------------------------------
# 4) Paramètres globaux
# ---------------------------------------------------------------------

# IDs Google Drive pour vos rasters
PH_FILE_ID  = "1-7go3Vp0q3AWKkVce9zeEEgbjdUUnwg7"  # pH final 3857
CLC_FILE_ID = "1whOuyOGM0dWea0K-cFgEXmPIaDlG489I"  # CLC final 3857

# Chemins des rasters en local
PH_FINAL  = "./ph_final_3857.tif"
CLC_FINAL = "./clc_final_3857.tif"

# Fichier HTML cèpes
CEPE_HTML = "./ph_veg_cepes.html"

# Paramètres cèpes
PH_MIN, PH_MAX = 5.0, 7.0
CLC_CEPES_CODES = [23, 24, 25, 29]

# Météo
METEO_DATA_DIR    = "./meteo_data"
METEO_RASTER_DIR  = "./meteo_rasters"
os.makedirs(METEO_DATA_DIR, exist_ok=True)
os.makedirs(METEO_RASTER_DIR, exist_ok=True)

DAYS_BEFORE = 6  # Nombre de jours avant j0 pour la fenêtre glissante
DAYS_AFTER  = 7  # Nombre de jours après j0
TODAY       = datetime.date.today()

# Variables Open-Meteo
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
PRECIP_MIN = 5.0
TEMP_MIN   = 5.0
TEMP_MAX   = 35.0
WIND_MAX   = 25.0

# Grille France
LAT_MIN, LAT_MAX = 41.0, 51.0
LON_MIN, LON_MAX = -5.0, 10.0
LAT_STEP, LON_STEP = 2.0, 2.0

# ---------------------------------------------------------------------
# 5) Fonctions utilitaires
# ---------------------------------------------------------------------

import math

def skip_if_exists(fpath):
    """Renvoie False si le fichier existe déjà => on skip la recréation."""
    if os.path.exists(fpath):
        print(f"[SKIP] {fpath} existe déjà => on ne le recrée pas.")
        return False
    return True

def log_raster_info(raster_path, description="Raster"):
    """Affiche des informations sur le raster."""
    try:
        with rasterio.open(raster_path) as ds:
            print(f"--- {description} ---")
            print(f"CRS: {ds.crs}")
            print(f"Extent: {ds.bounds}")
            print(f"Resolution: {ds.res}")
            print(f"Width: {ds.width}, Height: {ds.height}")
            print(f"Transform: {ds.transform}")
            print("-------------------------\n")
    except Exception as e:
        print(f"[ERROR] Impossible d'ouvrir le raster {raster_path}: {e}")

def create_france_grid(lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    lat_vals = np.arange(lat_min, lat_max + 0.0001, lat_step)
    lon_vals = np.arange(lon_min, lon_max + 0.0001, lon_step)
    points = []
    for la in lat_vals:
        for lo in lon_vals:
            points.append((la, lo))
    return points

# ---------------------------------------------------------------------
# 6) Téléchargement Météo : Grille France 2°
# ---------------------------------------------------------------------

def download_weather_xml(day):
    """
    Télécharge la météo pour une journée spécifique et la stocke en XML.
    """
    xml_path = os.path.join(METEO_DATA_DIR, f"meteo_{day}.xml")
    if os.path.exists(xml_path):
        print(f"[SKIP] {xml_path} existe déjà => on ne le retélécharge pas.")
        return xml_path

    print(f"[METEO] Téléchargement météo pour {day} => {xml_path}")
    pts = create_france_grid(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, LAT_STEP, LON_STEP)
    root = ET.Element("WeatherData", date=day.isoformat())

    for (la, lo) in tqdm(pts, desc=f"Téléchargement points météo {day}", ncols=80):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": la,
            "longitude": lo,
            "daily": DAILY_VARS,
            "start_date": day.isoformat(),
            "end_date":   day.isoformat(),
            "timezone": "auto"
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            daily = data.get("daily", {})
            time_list = daily.get("time", [])
            if not time_list:
                continue
            index = 0
            if index >= len(time_list):
                continue
            weather_point = ET.SubElement(root, "Point", latitude=str(la), longitude=str(lo))
            for var in DAILY_VARS.split(","):
                value = daily.get(var, [None])[index]
                ET.SubElement(weather_point, var).text = str(value) if value is not None else "NaN"
        except Exception as e:
            print(f"[WARN] Échec pour lat={la}, lon={lo}: {e}")

    try:
        tree = ET.ElementTree(root)
        tree.write(xml_path)
        print(f"[OK] XML => {xml_path}")
        return xml_path
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'écriture du fichier XML {xml_path}: {e}")
        return None

def interpolate_weather(xml_path, ph_raster, var):
    """
    Interpole les données météo depuis le fichier XML vers la grille du raster pH.
    On suppose que ph_raster = pH_final, déjà en EPSG:3857.
    """
    if not os.path.exists(xml_path):
        print(f"[WARN] Fichier XML introuvable: {xml_path}")
        return None

    print(f"[INTERPOLATE] {var} depuis {xml_path}")
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Erreur lors de la lecture du fichier XML {xml_path}: {e}")
        return None

    lats, lons, vals = [], [], []
    for point in root.findall("Point"):
        lat = float(point.get("latitude"))
        lon = float(point.get("longitude"))
        val_text = point.find(var).text
        try:
            val = float(val_text)
            if np.isnan(val):
                continue
        except:
            continue
        lats.append(lat)
        lons.append(lon)
        vals.append(val)

    if not lats:
        print(f"[WARN] Aucune donnée valide pour {var} dans {xml_path}")
        return None

    lats = np.array(lats)
    lons = np.array(lons)
    vals = np.array(vals)

    try:
        with rasterio.open(ph_raster) as ds:
            w, h = ds.width, ds.height
            lb, bb, rb, tb = ds.bounds
            lon_min, lat_min, lon_max, lat_max = transform_bounds(ds.crs, "EPSG:4326", lb, bb, rb, tb)
        gx = np.linspace(lon_min, lon_max, w)
        gy = np.linspace(lat_max, lat_min, h)
    except Exception as e:
        print(f"[ERROR] Impossible d'ouvrir le raster pH {ph_raster}: {e}")
        return None

    # Interpolation IDW via cKDTree
    from scipy.spatial import cKDTree
    tree_kd = cKDTree(np.c_[lats, lons])
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_points = np.c_[grid_y.ravel(), grid_x.ravel()]
    distances, indices = tree_kd.query(grid_points, k=4, p=2, distance_upper_bound=10)

    weights = 1.0 / np.where(distances == 0, 1e-12, distances)**2
    weights[distances == np.inf] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        interpolated = np.sum(weights * vals[indices], axis=1) / np.sum(weights, axis=1)
    interpolated = interpolated.reshape((h, w))
    interpolated[np.isnan(interpolated)] = 0

    return interpolated.astype(np.float32)

# ---------------------------------------------------------------------
# 7) Génération TIF + HTML
# ---------------------------------------------------------------------

def build_meteo_jX(offset, current_day, ph_path, clc_path, tif_path, html_path):
    """
    Génère les fichiers TIF et HTML pour un offset (j0..j+7) basé
    sur DAYS_BEFORE jours de données (j-6..j).
    """
    print(f"[METEO] Traitement pour j{offset} => {tif_path}, {html_path}")
    window_start = current_day - datetime.timedelta(days=DAYS_BEFORE)
    weather_vars = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
    meteo_data = {var: [] for var in weather_vars}

    shape = None
    try:
        with rasterio.open(ph_path) as ds_ph:
            shape = (ds_ph.height, ds_ph.width)
    except Exception as e:
        print(f"[ERROR] Impossible d'ouvrir {ph_path}: {e}")
        return

    # Téléchargement / Interpolation sur la période j-DAYS_BEFORE..j
    for day_offset in range(DAYS_BEFORE + 1):
        day = window_start + datetime.timedelta(days=day_offset)
        xml_path = os.path.join(METEO_DATA_DIR, f"meteo_{day}.xml")
        if not os.path.exists(xml_path):
            download_weather_xml(day)

        for var in weather_vars:
            interpolated = interpolate_weather(xml_path, ph_path, var)
            if interpolated is not None:
                meteo_data[var].append(interpolated)
            else:
                meteo_data[var].append(np.full(shape, np.nan, dtype=np.float32))

    # Appliquer les masques météo
    mask_meteo = np.ones(shape, dtype=bool)
    for var in weather_vars:
        stacked = np.stack(meteo_data[var], axis=0)
        if var == "temperature_2m_max":
            mask_var = np.all(stacked <= TEMP_MAX, axis=0)
        elif var == "temperature_2m_min":
            mask_var = np.all(stacked >= TEMP_MIN, axis=0)
        elif var == "precipitation_sum":
            mask_var = np.all(stacked >= PRECIP_MIN, axis=0)
        elif var == "wind_speed_10m_max":
            mask_var = np.all(stacked <= WIND_MAX, axis=0)
        else:
            mask_var = np.ones(shape, dtype=bool)
        mask_meteo &= mask_var

    # Combiner avec pH + végétation
    try:
        with rasterio.open(ph_path) as ds_ph, rasterio.open(clc_path) as ds_clc:
            ph_data  = ds_ph.read(1)
            clc_data = ds_clc.read(1)
            pH_nodata = ds_ph.nodata

            mask_clc = np.isin(clc_data, CLC_CEPES_CODES)
            valid_ph = (ph_data != pH_nodata) & np.isfinite(ph_data) if pH_nodata is not None else np.isfinite(ph_data)
            in_range_ph = (ph_data >= PH_MIN) & (ph_data <= PH_MAX)
            mask_ph_cond = np.where(valid_ph, in_range_ph, True)

            mask_cepe = mask_clc & mask_ph_cond
    except Exception as e:
        print(f"[ERROR] Erreur de lecture pH/CLC: {e}")
        return

    if mask_cepe.shape != mask_meteo.shape:
        print("[ERROR] Dimensions cèpes vs météo différentes.")
        return

    final_mask = mask_cepe & mask_meteo
    final_data = final_mask.astype(np.uint8) * 255

    valid_pixels = np.sum(final_mask)
    total_pixels = final_mask.size
    percent = 100.0 * valid_pixels / total_pixels
    print(f"[INFO] Pixels valides: {valid_pixels}/{total_pixels} ({percent:.2f}%)")

    # Écriture TIF
    try:
        with rasterio.open(ph_path) as ds_ph:
            prof = ds_ph.profile.copy()
        prof.update({
            "count": 1,
            "dtype": "uint8",
            "nodata": 0
        })
        with rasterio.open(tif_path, "w", **prof) as dst:
            dst.write(final_data[np.newaxis, :, :])
        print(f"[OK] TIF => {tif_path}")
        log_raster_info(tif_path, "Météo Combined TIF")
    except Exception as e:
        print(f"[ERROR] Erreur écriture TIF {tif_path}: {e}")

    # Écriture HTML
    try:
        with rasterio.open(tif_path) as ds_tif:
            lb, bb, rb, tb = ds_tif.bounds
            arr = ds_tif.read(1)
            lon_min, lat_min, lon_max, lat_max = transform_bounds(ds_tif.crs, "EPSG:4326", lb, bb, rb, tb)

        folium_bounds = [[lat_min, lon_min], [lat_max, lon_max]]
        c_lat = (lat_min + lat_max) / 2
        c_lon = (lon_min + lon_max) / 2

        arr_norm = arr / 255.0
        m = folium.Map(location=[c_lat, c_lon], zoom_start=6, tiles="OpenStreetMap")
        folium.raster_layers.ImageOverlay(
            image=arr_norm,
            bounds=folium_bounds,
            opacity=0.5,
            colormap=lambda x: (0, 1, 0, x),
            name=f"Météo7 j{offset}"
        ).add_to(m)
        m.save(html_path)
        print(f"[OK] HTML => {html_path}")
    except Exception as e:
        print(f"[ERROR] Erreur création HTML {html_path}: {e}")

# ---------------------------------------------------------------------
# 8) Upload GitHub (optionnel)
# ---------------------------------------------------------------------

def github_upload_file(local_path,
                       repo_owner=REPO_OWNER,
                       repo_name=REPO_NAME,
                       token=GITHUB_TOKEN,
                       commit_message="Upload via script",
                       remote_path=""):
    """Upload local_path sur le repo GitHub dans remote_path."""
    if not os.path.exists(local_path):
        print(f"[WARN] Fichier introuvable: {local_path}")
        return
    if not remote_path:
        remote_path = os.path.basename(local_path)

    if not token:
        print("[WARN] Pas de token GitHub => upload ignoré.")
        return

    print(f"[GitHub] Upload '{local_path}' => '{repo_owner}/{repo_name}' dans '{remote_path}' ...")
    try:
        with open(local_path, "rb") as f:
            content = f.read()
        content_b64 = base64.b64encode(content).decode("utf-8")

        get_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{remote_path}"
        headers = {
            "Authorization": f"token {token}",
            "Content-Type": "application/json"
        }
        get_resp = requests.get(get_url, headers=headers)
        if get_resp.status_code == 200:
            sha = get_resp.json().get("sha")
        else:
            sha = None

        data = {
            "message": commit_message,
            "content": content_b64,
        }
        if sha:
            data["sha"] = sha

        put_resp = requests.put(get_url, headers=headers, json=data)
        if put_resp.status_code in (200, 201):
            print("[GitHub] Upload réussi !")
        else:
            print(f"[GitHub][ERREUR] code={put_resp.status_code}")
            print("Réponse:", put_resp.text)
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'upload GitHub du fichier {local_path}: {e}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("[MAIN] Début du script Météo (GitHub)\n")
    if GITHUB_TOKEN:
        print(f"[DEBUG] GITHUB_TOKEN = {GITHUB_TOKEN[:5]}...")
    else:
        print("[WARN] GITHUB_TOKEN non défini.")

    # 1) Télécharger les rasters volumineux depuis Google Drive
    download_from_gdrive(PH_FILE_ID, PH_FINAL)
    download_from_gdrive(CLC_FILE_ID, CLC_FINAL)

    # 2) Vérifier la présence des rasters
    if not os.path.exists(PH_FINAL) or not os.path.exists(CLC_FINAL):
        print("[ERROR] Rasters introuvables après téléchargement.")
        return

    # 3) Génération j0..j+7
    total_offsets = DAYS_AFTER + 1
    for offset in range(total_offsets):
        current_day = TODAY + datetime.timedelta(days=offset)
        tif_path  = os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.tif")
        html_path = os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.html")
        build_meteo_jX(offset, current_day, PH_FINAL, CLC_FINAL, tif_path, html_path)

    # 4) (Optionnel) Uploader les fichiers TIF/HTML
    for offset in range(total_offsets):
        hpath = os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.html")
        tpath = os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.tif")
        if os.path.exists(hpath):
            github_upload_file(
                local_path=hpath,
                commit_message=f"Update meteo_j{offset}.html",
                remote_path=f"meteo_j{offset}.html"
            )
        if os.path.exists(tpath):
            github_upload_file(
                local_path=tpath,
                commit_message=f"Update meteo_j{offset}.tif",
                remote_path=f"meteo_j{offset}.tif"
            )

    print("[MAIN] Fin du script Météo (GitHub).\n")

if __name__ == "__main__":
    main()
