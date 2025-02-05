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
from bs4 import BeautifulSoup
import subprocess

# ---------------------------------------------------------------------
# 1) Gestion Google Drive (HTML form parsing)
# ---------------------------------------------------------------------

def _save_binary_response(response, destination):
    """Écrit la réponse HTTP binaire dans un fichier local, par chunks."""
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def download_gdrive_with_html_form(file_id, destination):
    """
    Télécharge un fichier volumineux depuis Google Drive,
    même s'il y a la page 'can't scan for viruses'.
    """
    import requests
    session = requests.Session()
    url = "https://docs.google.com/uc"
    params = {"export": "download", "id": file_id}

    r1 = session.get(url, params=params, stream=True)
    content_type = r1.headers.get("Content-Type", "").lower()

    if "text/html" not in content_type:
        # Pas de page intermédiaire => on écrit directement le flux
        _save_binary_response(r1, destination)
        return

    # Sinon, on parse la page HTML pour le fameux formulaire "can't scan"
    soup = BeautifulSoup(r1.text, "html.parser")
    form = soup.find("form")
    if not form:
        raise Exception("Impossible de trouver le <form> dans la page Google Drive.")
    action_url = form.get("action")
    if not action_url:
        raise Exception("Impossible de trouver l'action du formulaire Google Drive.")
    inputs = form.find_all("input")
    form_params = {}
    for inp in inputs:
        name = inp.get("name")
        value = inp.get("value")
        if name and value:
            form_params[name] = value

    r2 = session.get(action_url, params=form_params, stream=True)
    _save_binary_response(r2, destination)

def download_from_gdrive(file_id, local_path):
    """Télécharge un fichier depuis GDrive (si pas déjà local)."""
    if os.path.exists(local_path):
        print(f"[SKIP] {local_path} existe déjà.")
        return
    print(f"[DOWNLOAD] GoogleDrive ID={file_id} => {local_path}")
    try:
        download_gdrive_with_html_form(file_id, local_path)
        print(f"[OK] Téléchargé : {local_path}")
    except Exception as e:
        print(f"[ERROR] Échec téléchargement GDrive {file_id}: {e}")
        raise  # remonter l'exception

# ---------------------------------------------------------------------
# 2) Chargement variables d'environnement
# ---------------------------------------------------------------------

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
REPO_OWNER   = "Uncl3b3ns"
REPO_NAME    = "Cueillette-"

# ---------------------------------------------------------------------
# 3) Paramètres globaux
# ---------------------------------------------------------------------

PH_FILE_ID  = "1-7go3Vp0q3AWKkVce9zeEEgbjdUUnwg7"
CLC_FILE_ID = "1whOuyOGM0dWea0K-cFgEXmPIaDlG489I"

PH_FINAL  = "./ph_final_3857.tif"
CLC_FINAL = "./clc_final_3857.tif"

CEPE_HTML = "./ph_veg_cepes.html"

PH_MIN, PH_MAX = 5.0, 7.0
CLC_CEPES_CODES = [23, 24, 25, 29]

METEO_DATA_DIR    = "./meteo_data"
METEO_RASTER_DIR  = "./meteo_rasters"
os.makedirs(METEO_DATA_DIR, exist_ok=True)
os.makedirs(METEO_RASTER_DIR, exist_ok=True)

DAYS_BEFORE = 6   # j-6..j
DAYS_AFTER  = 7   # j0..j+7
TODAY       = datetime.date.today()

DAILY_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"

PRECIP_SUM_MIN = 20.0
TEMP_MIN   = 5.0
TEMP_MAX   = 35.0
WIND_MAX   = 25.0

LAT_MIN, LAT_MAX = 41.0, 51.0
LON_MIN, LON_MAX = -5.0, 10.0
LAT_STEP, LON_STEP = 2.0, 2.0

import requests

# ---------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------

def skip_if_exists(fpath):
    if os.path.exists(fpath):
        print(f"[SKIP] {fpath} => déjà présent.")
        return False
    return True

import xml.etree.ElementTree as ET
from tqdm import tqdm
import rasterio
from rasterio.warp import transform_bounds
from scipy.spatial import cKDTree

def create_france_grid(lat_min, lat_max, lon_min, lon_max, lat_step, lon_step):
    import numpy as np
    lat_vals = np.arange(lat_min, lat_max+0.0001, lat_step)
    lon_vals = np.arange(lon_min, lon_max+0.0001, lon_step)
    points=[]
    for la in lat_vals:
        for lo in lon_vals:
            points.append((la,lo))
    return points

# ---------------------------------------------------------------------
# 4) Téléchargement Météo
# ---------------------------------------------------------------------

def download_weather_xml(day):
    xml_path = os.path.join(METEO_DATA_DIR, f"meteo_{day}.xml")
    if os.path.exists(xml_path):
        print(f"[SKIP] {xml_path} existe déjà.")
        return xml_path

    print(f"[METEO] Download {day} => {xml_path}")
    pts = create_france_grid(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, LAT_STEP, LON_STEP)
    root = ET.Element("WeatherData", date=day.isoformat())

    for (la, lo) in tqdm(pts, desc=f"Météo {day}", ncols=80):
        url = "https://api.open-meteo.com/v1/forecast"
        params={
            "latitude": la,
            "longitude":lo,
            "daily":DAILY_VARS,
            "start_date": day.isoformat(),
            "end_date":   day.isoformat(),
            "timezone": "auto"
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data=r.json()
            daily=data.get("daily",{})
            time_list = daily.get("time",[])
            if not time_list:
                continue
            val_point = ET.SubElement(root, "Point", latitude=str(la), longitude=str(lo))
            for var in DAILY_VARS.split(","):
                v = daily.get(var,[None])[0]
                val_point.set(var, str(v) if v is not None else "NaN")
        except Exception as e:
            print(f"[WARN] lat={la},lon={lo}: {e}")

    try:
        tree=ET.ElementTree(root)
        tree.write(xml_path)
        print(f"[OK] => {xml_path}")
        return xml_path
    except Exception as e:
        print(f"[ERROR] Ecriture XML {xml_path}: {e}")
        return None

def interpolate_weather(xml_path, ph_raster, var):
    if not os.path.exists(xml_path):
        print(f"[WARN] Pas de fichier XML: {xml_path}")
        return None
    print(f"[INTERPOLATE] {var} -> {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERROR] Lecture XML {xml_path}: {e}")
        return None

    lats,lons,vals = [],[],[]
    for point in root.findall("Point"):
        lat = float(point.get("latitude"))
        lon = float(point.get("longitude"))
        val_text = point.get(var,"NaN")
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
        print(f"[WARN] Aucune donnée {var} dans {xml_path}")
        return None

    try:
        with rasterio.open(ph_raster) as ds_ph:
            w,h= ds_ph.width, ds_ph.height
            lb,bb,rb,tb= ds_ph.bounds
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                ds_ph.crs,"EPSG:4326",lb,bb,rb,tb)
        gx= np.linspace(lon_min,lon_max,w)
        gy= np.linspace(lat_max,lat_min,h)
    except Exception as e:
        print(f"[ERROR] Ouverture pH {ph_raster}: {e}")
        return None

    tree_kd = cKDTree(np.c_[lats,lons])
    grid_x, grid_y = np.meshgrid(gx,gy)
    grid_points= np.c_[grid_y.ravel(), grid_x.ravel()]
    distances,indices = tree_kd.query(grid_points, k=4, p=2, distance_upper_bound=10)

    weights = 1.0 / np.where(distances==0,1e-12,distances)**2
    weights[distances==np.inf]=0
    with np.errstate(divide='ignore',invalid='ignore'):
        arr_interp = np.sum(weights * np.array(vals)[indices],axis=1) / np.sum(weights,axis=1)
    arr_interp= arr_interp.reshape((h,w))
    arr_interp[np.isnan(arr_interp)] = 0
    return arr_interp.astype(np.float32)

# ---------------------------------------------------------------------
# 5) Carte pH+Vég
# ---------------------------------------------------------------------

def create_cepes_html(ph_path, clc_path, out_html):
    if os.path.exists(out_html):
        print(f"[SKIP] {out_html} => déjà présent.")
        return
    print(f"[CEPE] => {out_html}")

    try:
        with rasterio.open(ph_path) as ds_ph, rasterio.open(clc_path) as ds_clc:
            ph_data= ds_ph.read(1)
            clc_data= ds_clc.read(1)
            pH_nodata=ds_ph.nodata

            # Vegetation
            mask_clc = np.isin(clc_data, CLC_CEPES_CODES)
            # pH
            if pH_nodata is not None:
                valid_ph = (ph_data!=pH_nodata)&np.isfinite(ph_data)
            else:
                valid_ph = np.isfinite(ph_data)
            in_range_ph = (ph_data>=PH_MIN)&(ph_data<=PH_MAX)
            mask_ph_cond = np.where(valid_ph, in_range_ph, True)

            mask_final= mask_clc & mask_ph_cond
            lb,bb,rb,tb=ds_ph.bounds
            from rasterio.warp import transform_bounds
            lon_min, lat_min, lon_max, lat_max= transform_bounds(
                ds_ph.crs,"EPSG:4326",lb,bb,rb,tb
            )
        overlay = (mask_final.astype(np.uint8)*255)
        folium_bounds= [[lat_min,lon_min],[lat_max,lon_max]]
        c_lat= (lat_min+lat_max)/2
        c_lon= (lon_min+lon_max)/2

        m= folium.Map(location=[c_lat,c_lon], zoom_start=6, tiles="OpenStreetMap")
        folium.raster_layers.ImageOverlay(
            image=overlay,
            bounds=folium_bounds,
            opacity=0.5,
            colormap=lambda x:(1,0,0,x),
            name="Cepes"
        ).add_to(m)
        m.save(out_html)
        print(f"[OK] => {out_html}")
    except Exception as e:
        print(f"[ERROR] create_cepes_html: {e}")
        # Si erreur => créer un fichier minimal "Données indisponibles"
        with open(out_html,"w",encoding="utf-8") as f:
            f.write("""
<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><title>Erreur</title></head>
<body>
<h2 style="color:red;text-align:center;">Données non disponibles (erreur cepe)</h2>
</body>
</html>
""")

# ---------------------------------------------------------------------
# 6) Génération HTML météo
# ---------------------------------------------------------------------

def build_meteo_jX(offset, current_day, ph_path, clc_path, out_html):
    """Crée un fichier HTML (sans TIF) pour jX."""
    if os.path.exists(out_html):
        print(f"[SKIP] {out_html} => déjà présent.")
        return

    print(f"[METEO] j{offset} => {out_html}")
    window_start= current_day - datetime.timedelta(days=DAYS_BEFORE)
    weather_vars= ["temperature_2m_max","temperature_2m_min","precipitation_sum","wind_speed_10m_max"]
    meteo_data={var:[] for var in weather_vars}

    shape=None
    try:
        with rasterio.open(ph_path) as ds_ph:
            shape=(ds_ph.height, ds_ph.width)
    except Exception as e:
        print(f"[ERROR] Ouverture pH {ph_path}: {e}")
        # Si ouverture pH impossible => noData
        with open(out_html,"w",encoding="utf-8") as f:
            f.write("""
<html><body><h2 style="color:red;">Données non disponibles (erreur pH)</h2></body></html>
""")
        return

    # Téléchargement / Interpolation j-6..j
    error_occurred=False
    for day_offset in range(DAYS_BEFORE+1):
        day= window_start + datetime.timedelta(days=day_offset)
        xml_path= os.path.join(METEO_DATA_DIR,f"meteo_{day}.xml")
        if not os.path.exists(xml_path):
            try:
                download_weather_xml(day)
            except Exception as e:
                print(f"[ERROR] Météo j{offset}, day={day}: {e}")
                error_occurred=True

        for var in weather_vars:
            if error_occurred:
                # On ne tente plus d'interpoler si on sait qu'il y a eu une erreur
                meteo_data[var].append(np.full(shape, np.nan, dtype=np.float32))
                continue
            arr= interpolate_weather(xml_path, ph_path, var)
            if arr is not None:
                meteo_data[var].append(arr)
            else:
                # Indique une erreur pour cette variable => tout le jour est noData
                error_occurred=True
                meteo_data[var].append(np.full(shape, np.nan, dtype=np.float32))

    if error_occurred:
        # Produire un fichier minimal "No Data"
        with open(out_html,"w",encoding="utf-8") as f:
            f.write(f"""
<html lang="fr"><head><meta charset="UTF-8"><title>Météo j{offset}</title></head>
<body>
<h2 style="color:red;text-align:center;">
Données météorologiques indisponibles pour j{offset}
</h2>
</body></html>
""")
        print(f"[INFO] Erreur => No data j{offset}")
        return

    # Sinon, on calcule le masque météo
    mask_meteo= np.ones(shape, dtype=bool)
    for var in weather_vars:
        stacked = np.stack(meteo_data[var], axis=0)
        if var=="temperature_2m_max":
            mask_var= np.all(stacked<=TEMP_MAX, axis=0)
        elif var=="temperature_2m_min":
            mask_var= np.all(stacked>=TEMP_MIN, axis=0)
        elif var=="precipitation_sum":
            sum_precip= np.sum(stacked, axis=0)
            mask_var= sum_precip>=PRECIP_SUM_MIN
        elif var=="wind_speed_10m_max":
            mask_var= np.all(stacked<=WIND_MAX, axis=0)
        else:
            mask_var= np.ones(shape,dtype=bool)
        mask_meteo &= mask_var

    # pH+Vég
    try:
        with rasterio.open(ph_path) as ds_ph, rasterio.open(clc_path) as ds_clc:
            ph_data= ds_ph.read(1)
            clc_data= ds_clc.read(1)
            pH_nodata= ds_ph.nodata
            mask_clc= np.isin(clc_data, CLC_CEPES_CODES)
            if pH_nodata is not None:
                valid_ph= (ph_data!=pH_nodata)&np.isfinite(ph_data)
            else:
                valid_ph= np.isfinite(ph_data)
            in_range_ph= (ph_data>=PH_MIN)&(ph_data<=PH_MAX)
            mask_ph_cond= np.where(valid_ph, in_range_ph, True)
            mask_cepe= mask_clc & mask_ph_cond
    except Exception as e:
        print(f"[ERROR] Lecture pH/CLC j{offset}: {e}")
        with open(out_html,"w",encoding="utf-8") as f:
            f.write(f"<html><body><h2 style='color:red'>Erreur pH/CLC j{offset}</h2></body></html>")
        return

    if mask_cepe.shape!=mask_meteo.shape:
        print("[ERROR] Dimensions mismatch pH/CLC vs météo.")
        with open(out_html,"w",encoding="utf-8") as f:
            f.write(f"<html><body><h2>Dimension mismatch j{offset}</h2></body></html>")
        return

    final_mask= mask_cepe & mask_meteo
    final_data= final_mask.astype(np.uint8)*255
    valid_pixels= np.sum(final_mask)
    total_pixels= final_mask.size
    prc= (valid_pixels/total_pixels)*100
    print(f"[INFO] j{offset}: {valid_pixels}/{total_pixels} => {prc:.2f}% propice")

    # Génération HTML Folium, sans marker
    try:
        arr_norm= final_data/255.0
        with rasterio.open(ph_path) as ds_tif:
            lb,bb,rb,tb= ds_tif.bounds
            lon_min, lat_min, lon_max, lat_max= transform_bounds(
                ds_tif.crs,"EPSG:4326",lb,bb,rb,tb
            )
        folium_bounds= [[lat_min,lon_min],[lat_max,lon_max]]
        c_lat= (lat_min+lat_max)/2
        c_lon= (lon_min+lon_max)/2

        m= folium.Map(location=[c_lat,c_lon], zoom_start=6, tiles="OpenStreetMap")
        folium.raster_layers.ImageOverlay(
            image=arr_norm,
            bounds=folium_bounds,
            opacity=0.5,
            colormap=lambda x:(0,1,0,x),
            name=f"meteo_j{offset}"
        ).add_to(m)

        # On ajoute un DIV en bas de page pour informer du nombre de pixels
        m.get_root().html.add_child(folium.Element(f"""
            <div style="text-align:center;padding:10px;font-weight:bold;">
                Zones valides : {valid_pixels}/{total_pixels} ({prc:.2f}%)
            </div>
        """))

        m.save(out_html)
        print(f"[OK] => {out_html}")
    except Exception as e:
        print(f"[ERROR] Final HTML j{offset}: {e}")
        with open(out_html,"w",encoding="utf-8") as f:
            f.write(f"<html><body><h2>Erreur finale j{offset}</h2></body></html>")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    print("[MAIN] Début du script Météo + Single Commit\n")
    if len(GITHUB_TOKEN)<10:
        print("[WARN] GITHUB_TOKEN non défini ou trop court => impossible push ?")

    # 1) Téléchargement pH+CLC
    try:
        download_from_gdrive(PH_FILE_ID, PH_FINAL)
        download_from_gdrive(CLC_FILE_ID, CLC_FINAL)
    except Exception as e:
        print(f"[ERROR] Téléchargement ph/clc: {e}")
        return

    if not os.path.exists(PH_FINAL) or not os.path.exists(CLC_FINAL):
        print("[ERROR] Rasters introuvables => stop.")
        return

    # 2) Créer carte pH+Veg => ph_veg_cepes.html
    create_cepes_html(PH_FINAL, CLC_FINAL, CEPE_HTML)

    # 3) Générer météo j0..j+7
    total_offsets= DAYS_AFTER+1  # 8
    for offset in range(total_offsets):
        cday= TODAY + datetime.timedelta(days=offset)
        html_path= os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.html")
        build_meteo_jX(offset, cday, PH_FINAL, CLC_FINAL, html_path)

    # 4) Single Commit => un seul build GitHub Pages
    # On regroupe tous les HTML créés
    html_paths= []
    if os.path.exists(CEPE_HTML):
        html_paths.append(CEPE_HTML)

    for offset in range(total_offsets):
        mp= os.path.join(METEO_RASTER_DIR, f"meteo_j{offset}.html")
        if os.path.exists(mp):
            html_paths.append(mp)

    if not html_paths:
        print("[INFO] Aucun fichier HTML => pas de commit.")
        return

    print(f"[INFO] Commit unique pour {len(html_paths)} fichiers HTML.")
    commit_msg= f"[AUTO] Update meteo {datetime.datetime.now().isoformat()}"
    # Config user
    os.system("git config user.email 'github-actions@github.com'")
    os.system("git config user.name 'GitHub Actions'")
    # Add
    to_add= " ".join(f'"{p}"' for p in html_paths)
    os.system(f'git add {to_add}')
    # Commit
    ret= os.system(f'git commit -m "{commit_msg}"')
    if ret!=0:
        print("[WARN] Rien à committer ? ou commit error.")
    else:
        # Push
        os.system("git push")
        print("[OK] Push fait => un seul build GitHub Pages va se lancer.")

    print("[MAIN] Fin du script.\n")

if __name__=="__main__":
    main()
