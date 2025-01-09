import os
import datetime
import numpy as np
import requests
import rasterio
from rasterio.warp import (
    calculate_default_transform, reproject, transform_bounds, Resampling
)
import folium
import base64
from tqdm import tqdm
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree
from rasterio.crs import CRS
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_OWNER   = "Uncl3b3ns"
REPO_NAME    = "Cueillette-" 

# ---------------------------------------------------------------------
# PARAMÈTRES GLOBAUX
# ---------------------------------------------------------------------

# Rasters bruts sans CRS assigné
PH_RASTER_RAW  = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\250_GSNmap_mean_ph_0_30.tif"
CLC_RASTER_RAW = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\U2018_CLC2018_V2020_20u1.tif"

# Définissez le CRS correct des rasters sources
PH_SOURCE_CRS  = "EPSG:4326"  # Pour pH, WGS84
CLC_SOURCE_CRS = "EPSG:4326"  # Pour végétation, WGS84

# Rasters reprojetés/finaux
PH_3857        = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\ph_3857.tif"
CLC_3857       = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\clc_3857.tif"
CLC_250m_3857  = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\clc_250m_3857.tif"
PH_250m_3857   = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\ph_250m_3857.tif"
PH_FINAL       = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\ph_final_3857.tif"
CLC_FINAL      = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\clc_final_3857.tif"

# Fichier HTML cèpes
CEPE_HTML = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\ph_veg_cepes.html"

# Paramètres cèpes
PH_MIN, PH_MAX = 5.0, 7.0
# S'il n'y a pas de valeur pH sur un pixel => on ignore le filtre pH
# On filtre juste la végétation.
# CLC: ex. broadleaf(23=311), conifer(24=312), mixed(25=313), transitional(29=324)
CLC_CEPES_CODES = [23, 24, 25, 29]

# Météo
METEO_DATA_DIR    = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\meteo_data"
METEO_RASTER_DIR  = r"C:\Users\guill\OneDrive\Desktop\Données Qgis\Projet champignons\meteo_rasters"
os.makedirs(METEO_DATA_DIR, exist_ok=True)
os.makedirs(METEO_RASTER_DIR, exist_ok=True)

DAYS_BEFORE = 6  # Nombre de jours avant j0 pour la fenêtre glissante
DAYS_AFTER  = 7  # Nombre de jours après j0 (j+1 à j+7)
TODAY       = datetime.date.today()

# Variables Open-Meteo
DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
PRECIP_MIN = 5.0
TEMP_MIN   = 5.0
TEMP_MAX   = 35.0
WIND_MAX   = 25.0

# Grille France : espacement 2° pour limiter le temps
LAT_MIN, LAT_MAX = 41.0, 51.0
LON_MIN, LON_MAX = -5.0, 10.0
LAT_STEP, LON_STEP = 2.0, 2.0  # plus large pour accélérer

# ---------------------------------------------------------------------
# 1) Fonctions Utilitaires
# ---------------------------------------------------------------------

def skip_if_exists(fpath):
    """Renvoie False si le fichier existe déjà => on skip la recréation."""
    if os.path.exists(fpath):
        print(f"[SKIP] {fpath} existe déjà => on ne le recrée pas.")
        return False
    return True

def safe_delete(path):
    """Forcer la suppression si besoin."""
    if os.path.exists(path):
        os.remove(path)
        print(f"[DELETE] {path}")

def log_raster_info(raster_path, description="Raster"):
    """Affiche des informations sur le raster."""
    try:
        with rasterio.open(raster_path) as ds:
            print(f"--- {description} ---")
            print(f"CRS: {ds.crs}")
            print(f"Extent: {ds.bounds}")
            print(f"Resolution: {ds.res}")  # Utilisez 'res' au lieu de 'resolution'
            print(f"Width: {ds.width}, Height: {ds.height}")
            print(f"Transform: {ds.transform}")
            print("-------------------------\n")
    except Exception as e:
        print(f"[ERROR] Impossible d'ouvrir le raster {raster_path}: {e}")

# ---------------------------------------------------------------------
# 2) Assignation du CRS si Nécessaire
# ---------------------------------------------------------------------

def assign_crs(raster_path, crs_epsg):
    """
    Assigne un CRS à un raster existant si celui-ci n'en a pas.
    Retourne le chemin du raster avec CRS assigné.
    """
    try:
        with rasterio.open(raster_path) as src:
            if src.crs:
                print(f"[INFO] Le raster {raster_path} a déjà un CRS: {src.crs}")
                return raster_path  # Pas besoin de réassigner
            else:
                print(f"[INFO] Assignation du CRS EPSG:{crs_epsg} au raster {raster_path}")
    except Exception as e:
        print(f"[ERROR] Impossible d'ouvrir le raster {raster_path}: {e}")
        return None

    try:
        # Lire les métadonnées existantes
        with rasterio.open(raster_path) as src:
            meta = src.meta.copy()

        # Assigner le CRS
        meta['crs'] = CRS.from_epsg(crs_epsg)

        # Écrire un nouveau fichier avec le CRS assigné
        new_raster_path = os.path.splitext(raster_path)[0] + "_with_crs.tif"
        with rasterio.open(new_raster_path, 'w', **meta) as dst:
            dst.write(src.read())

        print(f"[OK] CRS EPSG:{crs_epsg} assigné et sauvegardé dans {new_raster_path}")
        log_raster_info(new_raster_path, "Assigned CRS Raster")
        return new_raster_path
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'assignation du CRS pour {raster_path}: {e}")
        return None

# ---------------------------------------------------------------------
# 3) Reprojection et Resampling
# ---------------------------------------------------------------------

def reproject_raster(src_path, dst_path, src_crs, resampling_method=Resampling.nearest):
    """Reprojette un raster vers EPSG:3857."""
    if os.path.exists(dst_path):
        print(f"[SKIP] {dst_path} existe déjà.")
        return

    print(f"[REPROJ] Reprojection de {src_path} vers {dst_path}")
    try:
        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src_crs, "EPSG:3857", src.width, src.height, *src.bounds
            )
            profile = src.meta.copy()
            profile.update({
                "crs": "EPSG:3857",
                "transform": transform,
                "width": width,
                "height": height
            })

            with rasterio.open(dst_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs="EPSG:3857",
                        resampling=resampling_method
                    )
        print(f"[OK] Reprojection terminée pour {dst_path}")
        log_raster_info(dst_path, "Reprojected Raster")
    except Exception as e:
        print(f"[ERROR] Erreur lors de la reprojection de {src_path} vers {dst_path}: {e}")

def resample_raster(src_path, dst_path, target_resolution, resampling_method=Resampling.nearest,
                   target_transform=None, target_width=None, target_height=None):
    """Resample un raster à une résolution cible."""
    if os.path.exists(dst_path):
        print(f"[SKIP] {dst_path} existe déjà.")
        return

    print(f"[RESAMPLE] Resampling de {src_path} vers {dst_path} avec une résolution de {target_resolution}m")
    try:
        with rasterio.open(src_path) as src:
            if target_transform and target_width and target_height:
                transform = target_transform
                width = target_width
                height = target_height
            else:
                transform, width, height = calculate_default_transform(
                    src.crs, src.crs, src.width, src.height, *src.bounds,
                    resolution=target_resolution
                )
            profile = src.meta.copy()
            profile.update({
                "transform": transform,
                "width": width,
                "height": height
            })

            with rasterio.open(dst_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=resampling_method
                    )
        print(f"[OK] Resampling terminé pour {dst_path}")
        log_raster_info(dst_path, "Resampled Raster")
    except Exception as e:
        print(f"[ERROR] Erreur lors du resampling de {src_path} vers {dst_path}: {e}")

# ---------------------------------------------------------------------
# 4) Copie de Raster
# ---------------------------------------------------------------------

def copy_raster(src_path, dst_path):
    """Copie brute si pas déjà exist."""
    if not skip_if_exists(dst_path):
        return
    print(f"[COPY] {src_path} => {dst_path}")
    try:
        with rasterio.open(src_path) as src:
            prof = src.meta.copy()
            dat = src.read()
        with rasterio.open(dst_path, "w", **prof) as dst:
            dst.write(dat)
        print("[OK] Copied")
        log_raster_info(dst_path, "Copied Raster")
    except Exception as e:
        print(f"[ERROR] Erreur lors de la copie de {src_path} vers {dst_path}: {e}")

# ---------------------------------------------------------------------
# 5) Carte Cèpes (pH + Végétation)
# ---------------------------------------------------------------------

def create_cepes_html(ph_path, clc_path, out_html):
    if not skip_if_exists(out_html):
        return
    print(f"[CEPEMAP] => {out_html}")

    try:
        with rasterio.open(ph_path) as ds_ph, rasterio.open(clc_path) as ds_clc:
            ph_data  = ds_ph.read(1)
            clc_data = ds_clc.read(1)
            # On récupère la nodata pH si exist
            pH_nodata = ds_ph.nodata

            # Végétation
            mask_clc = np.isin(clc_data, CLC_CEPES_CODES)

            # pH dans la fourchette
            if pH_nodata is not None:
                valid_ph = (ph_data != pH_nodata) & np.isfinite(ph_data)
            else:
                valid_ph = np.isfinite(ph_data)

            in_range_ph = (ph_data >= PH_MIN) & (ph_data <= PH_MAX)
            mask_ph_cond = np.where(valid_ph, in_range_ph, True)

            # Mask final
            mask_final = mask_clc & mask_ph_cond

            lb, bb, rb, tb = ds_ph.bounds
            lon_min, lat_min, lon_max, lat_max = transform_bounds(ds_ph.crs, "EPSG:4326", lb, bb, rb, tb)

        # Vérifier que les dimensions des rasters correspondent
        if mask_final.shape != mask_clc.shape:
            print(f"[ERROR] Les masques pH et CLC ont des formes différentes: {mask_final.shape} vs {mask_clc.shape}")
            return

        overlay = mask_final.astype(np.uint8) * 255
        folium_bounds = [[lat_min, lon_min], [lat_max, lon_max]]
        c_lat = (lat_min + lat_max) / 2
        c_lon = (lon_min + lon_max) / 2

        m = folium.Map(location=[c_lat, c_lon], zoom_start=6, tiles="OpenStreetMap")
        folium.raster_layers.ImageOverlay(
            image=overlay,
            bounds=folium_bounds,
            opacity=0.5,
            colormap=lambda x: (1, 0, 0, x),
            name="Cèpes pH+CLC"
        ).add_to(m)
        m.save(out_html)
        print(f"[OK] => {out_html}")
        # Note: Les fichiers HTML ne contiennent pas d'information CRS
    except Exception as e:
        print(f"[ERROR] Erreur lors de la création de la carte Cèpes: {e}")

# ---------------------------------------------------------------------
# 6) Téléchargement Météo : Grille France 2°
# ---------------------------------------------------------------------

def create_france_grid(lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    lat_vals = np.arange(lat_min, lat_max + 0.0001, lat_step)
    lon_vals = np.arange(lon_min, lon_max + 0.0001, lon_step)
    points = []
    for la in lat_vals:
        for lo in lon_vals:
            points.append((la, lo))
    return points

def download_weather_xml(day):
    """
    Télécharge la météo pour une journée spécifique et la stocke en XML.
    """
    xml_path = os.path.join(METEO_DATA_DIR, f"meteo_{day}.xml")
    if os.path.exists(xml_path):
        print(f"[SKIP] {xml_path} existe déjà => on ne le retélécharge pas.")
        return xml_path

    print(f"[METEO] Téléchargement météo pour {day} => {xml_path}")
    # Grille 2° => ~ 5x8 = 40 points
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
            index = 0  # Un seul jour
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
    Interpole les données météo depuis le fichier XML vers la grille du raster ph.
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

        # IDW Interpolation using cKDTree for performance
        tree_kd = cKDTree(np.c_[lats, lons])
        grid_x, grid_y = np.meshgrid(gx, gy)
        grid_points = np.c_[grid_y.ravel(), grid_x.ravel()]
        distances, indices = tree_kd.query(grid_points, k=4, p=2, distance_upper_bound=10)

        weights = 1 / np.where(distances == 0, 1e-12, distances) ** 2
        weights[distances == np.inf] = 0
        # Handle cases where all weights are zero to avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            interpolated = np.sum(weights * vals[indices], axis=1) / np.sum(weights, axis=1)
        interpolated = interpolated.reshape((h, w))
        interpolated[np.isnan(interpolated)] = 0  # Remplacer les NaN par 0 ou une autre valeur appropriée

        return interpolated.astype(np.float32)
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'interpolation des données météo: {e}")
        return None

# ---------------------------------------------------------------------
# 7) Création des fichiers TIF et HTML pour chaque offset j0 à j+7
# ---------------------------------------------------------------------

def build_meteo_jX(offset, current_day, ph_path, clc_path, tif_path, html_path):
    """
    Génère les fichiers TIF et HTML pour un offset spécifique basé sur une fenêtre de 7 jours.
    """
    print(f"[METEO] Traitement pour j{offset} ({current_day}) => {tif_path}, {html_path}")

    # Définir la fenêtre de 7 jours
    window_start = current_day - datetime.timedelta(days=DAYS_BEFORE)
    window_end = current_day

    # Télécharger les données météo pour la fenêtre
    weather_vars = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
    meteo_data = {var: [] for var in weather_vars}

    # Lire une seule fois la forme du raster pH pour éviter de l'ouvrir plusieurs fois
    try:
        with rasterio.open(ph_path) as ds_ph:
            shape = (ds_ph.height, ds_ph.width)
    except Exception as e:
        print(f"[ERROR] Impossible d'ouvrir le raster pH {ph_path}: {e}")
        return

    for day_offset in range(DAYS_BEFORE + 1):
        day = window_start + datetime.timedelta(days=day_offset)
        xml_path = os.path.join(METEO_DATA_DIR, f"meteo_{day}.xml")
        if not os.path.exists(xml_path):
            download_weather_xml(day)
        # Interpoler chaque variable
        for var in weather_vars:
            interpolated = interpolate_weather(xml_path, ph_path, var)
            if interpolated is not None:
                meteo_data[var].append(interpolated)
            else:
                # Si interpolation échoue, ajouter un tableau de NaN
                meteo_data[var].append(np.full(shape, np.nan, dtype=np.float32))

    # Appliquer les filtres sur la fenêtre glissante
    mask_meteo = np.ones(shape, dtype=bool)

    for var in weather_vars:
        # Stack des données pour les 7 jours
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

    # Combiner avec les couches pH et végétation
    try:
        with rasterio.open(ph_path) as ds_ph, rasterio.open(clc_path) as ds_clc:
            ph_data  = ds_ph.read(1)
            clc_data = ds_clc.read(1)
            pH_nodata = ds_ph.nodata

            # Végétation
            mask_clc = np.isin(clc_data, CLC_CEPES_CODES)

            # pH dans la fourchette
            if pH_nodata is not None:
                valid_ph = (ph_data != pH_nodata) & np.isfinite(ph_data)
            else:
                valid_ph = np.isfinite(ph_data)

            in_range_ph = (ph_data >= PH_MIN) & (ph_data <= PH_MAX)
            mask_ph_cond = np.where(valid_ph, in_range_ph, True)

            # Mask final
            mask_cepe = mask_clc & mask_ph_cond
    except Exception as e:
        print(f"[ERROR] Erreur lors de la combinaison des couches pH et végétation: {e}")
        return

    # Vérifier que les dimensions des rasters correspondent
    if mask_cepe.shape != mask_meteo.shape:
        print(f"[ERROR] Les masques Cèpes et Météo ont des formes différentes: {mask_cepe.shape} vs {mask_meteo.shape}")
        return

    # Combiner tous les masques
    final_mask = mask_cepe & mask_meteo
    final_data = final_mask.astype(np.uint8) * 255  # 255 pour les zones valides

    # Afficher le nombre de pixels valides
    valid_pixels = np.sum(final_mask)
    total_pixels = final_mask.size
    percentage = (valid_pixels / total_pixels) * 100
    print(f"[PROGRESS] Pixels valides: {valid_pixels}/{total_pixels} ({percentage:.2f}%)")

    # Écrire le fichier TIF
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
        print(f"[OK] => {tif_path}")
        log_raster_info(tif_path, "Météo Combined TIF")
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'écriture du fichier TIF {tif_path}: {e}")
        return

    # Générer le fichier HTML avec Folium
    try:
        with rasterio.open(tif_path) as ds_tif:
            lb, bb, rb, tb = ds_tif.bounds
            arr = ds_tif.read(1)
            lon_min, lat_min, lon_max, lat_max = transform_bounds(ds_tif.crs, "EPSG:4326", lb, bb, rb, tb)

        folium_bounds = [[lat_min, lon_min], [lat_max, lon_max]]
        c_lat = (lat_min + lat_max) / 2
        c_lon = (lon_min + lon_max) / 2

        # Normalize the data for visualization
        arr_norm = arr / 255.0  # Convert to 0-1 range for opacity

        m = folium.Map(location=[c_lat, c_lon], zoom_start=6, tiles="OpenStreetMap")
        folium.raster_layers.ImageOverlay(
            image=arr_norm,
            bounds=folium_bounds,
            opacity=0.5,
            colormap=lambda x: (0, 1, 0, x),
            name=f"Météo7 j{offset}"
        ).add_to(m)
        m.save(html_path)
        print(f"[OK] => {html_path}")
    except Exception as e:
        print(f"[ERROR] Erreur lors de la création du fichier HTML {html_path}: {e}")

# ---------------------------------------------------------------------
# 8) Upload GitHub
# ---------------------------------------------------------------------

def github_upload_file(local_path,
                       repo_owner=REPO_OWNER,
                       repo_name=REPO_NAME,
                       token=GITHUB_TOKEN,
                       commit_message="Upload via script",
                       remote_path=""):
    """Upload 'local_path' au repo GitHub dans 'remote_path'."""
    if not os.path.exists(local_path):
        print(f"[WARN] Fichier introuvable: {local_path}")
        return
    if not remote_path:
        remote_path = os.path.basename(local_path)

    print(f"[GitHub][DEBUG] Upload '{local_path}' => '{repo_owner}/{repo_name}' dans '{remote_path}'...")
    try:
        with open(local_path, "rb") as f:
            content = f.read()
        content_b64 = base64.b64encode(content).decode("utf-8")

        # Vérifier si le fichier existe déjà pour obtenir le SHA nécessaire
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
    print("[MAIN] Début du script\n")

    # 1) Assignation du CRS si nécessaire
    print("[MAIN] Assignation du CRS aux rasters sources si nécessaire.")
    ph_raster_assigned = assign_crs(PH_RASTER_RAW, 4326)
    clc_raster_assigned = assign_crs(CLC_RASTER_RAW, 4326)

    if ph_raster_assigned is None or clc_raster_assigned is None:
        print("[MAIN][ERROR] Assignation du CRS échouée. Arrêt du script.")
        return

    # 2) Reprojection en EPSG:3857
    print("[MAIN] Reprojection des rasters en EPSG:3857.")
    reproject_raster(
        src_path=ph_raster_assigned,
        dst_path=PH_3857,
        src_crs="EPSG:4326",
        resampling_method=Resampling.bilinear  # Pour pH (continu)
    )
    reproject_raster(
        src_path=clc_raster_assigned,
        dst_path=CLC_3857,
        src_crs="EPSG:4326",
        resampling_method=Resampling.nearest  # Pour végétation (catégoriel)
    )

    # 3) Resampling pour aligner les résolutions
    print("[MAIN] Resampling de la végétation pour correspondre à la résolution de 250m.")
    resample_raster(
        src_path=CLC_3857,
        dst_path=CLC_250m_3857,
        target_resolution=250,  # en mètres
        resampling_method=Resampling.nearest  # Pour données catégorielles
    )

    # Resampling pH à 250m en alignant la grille avec CLC
    print("[MAIN] Resampling du pH pour correspondre à la résolution de 250m et alignement avec CLC.")
    try:
        with rasterio.open(CLC_250m_3857) as ds_clc_250m:
            target_transform = ds_clc_250m.transform
            target_width = ds_clc_250m.width
            target_height = ds_clc_250m.height
    except Exception as e:
        print(f"[ERROR] Impossible d'ouvrir {CLC_250m_3857} pour obtenir les paramètres de resampling: {e}")
        return

    resample_raster(
        src_path=PH_3857,
        dst_path=PH_250m_3857,
        target_resolution=250,  # en mètres
        resampling_method=Resampling.bilinear,  # Pour pH (continu)
        target_transform=target_transform,
        target_width=target_width,
        target_height=target_height
    )

    # 4) Créer les rasters finaux alignés
    print("[MAIN] Copie des rasters finaux alignés.")
    copy_raster(PH_250m_3857, PH_FINAL)
    copy_raster(CLC_250m_3857, CLC_FINAL)

    # 5) Créer la carte Cèpes (pH + Végétation) une seule fois
    print("[MAIN] Création de la carte Cèpes (pH + Végétation).")
    create_cepes_html(PH_FINAL, CLC_FINAL, CEPE_HTML)

    # 6) Génération des cartes météo j0..j+7 (7 jours glissants)
    print("[MAIN] Génération des cartes météo j0..j+7 (7 jours glissants).")
    total_offsets = DAYS_AFTER + 1  # de j0 à j+7 inclus

    for offset in tqdm(range(total_offsets), desc="Cartes météo", ncols=80):
        current_day = TODAY + datetime.timedelta(days=offset)
        tif_path = os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.tif")
        html_path = os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.html")
        build_meteo_jX(offset, current_day, PH_FINAL, CLC_FINAL, tif_path, html_path)

    # 7) Upload GitHub
    print("[MAIN] Upload GitHub des fichiers HTML + TIF finaux...")
    # Carte cèpes => "cepes.html"
    if os.path.exists(CEPE_HTML):
        github_upload_file(
            local_path=CEPE_HTML,
            commit_message="Update cepes.html",
            remote_path="cepes.html"
        )

    # Cartes météo j0..j+7 => "meteo_j0.html", "meteo_j0.tif", etc.
    for offset in range(total_offsets):
        hpath = os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.html")
        tpath = os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.tif")
        if os.path.exists(hpath):
            remote_name = f"meteo_j{offset}.html"
            github_upload_file(
                local_path=hpath,
                commit_message=f"Update {remote_name}",
                remote_path=remote_name
            )
        if os.path.exists(tpath):
            remote_name = f"meteo_j{offset}.tif"
            github_upload_file(
                local_path=tpath,
                commit_message=f"Update {remote_name}",
                remote_path=remote_name
            )

    print("[MAIN] Fin du script.\n")

if __name__ == "__main__":
    main()
