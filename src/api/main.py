# =====================================================================
# 1. WORKAROUND WINDOWS APPLOCKER & ML DEADLOCK FIX
# =====================================================================
import os
# Memaksa pustaka ML numerik menggunakan 1 thread agar tidak terjadi deadlock
# saat dijalankan bersamaan dengan arsitektur asynchronous FastAPI
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Pustaka Machine Learning wajib diimpor SETELAH os.environ ditetapkan
import sklearn
import sklearn.tree
import sklearn.ensemble
import lightgbm
import xgboost
import joblib


# =====================================================================
# 2. STANDARD LIBRARY & THIRD-PARTY IMPORTS
# =====================================================================
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from .schemas import UserCreate, UserOut, CatchReportCreate, CatchReportOut
from .crud import create_user, create_catch_report, get_reports_by_location
from fastapi.middleware.cors import CORSMiddleware

# =====================================================================
# 3. LOCAL MODULE IMPORTS
# =====================================================================
from .database import engine, Base, get_db, AsyncSessionLocal
from .crud import (
    fetch_data,
    get_or_create_location,
    get_all_locations,
    get_location_by_name,
    save_marine_weather,
    get_marine_weather,
    save_prediction,
    get_prediction_history,
    get_prediction_by_id,
)
from .ml import generate_forecast, load_all_models
from .schemas import (
    LocationOut, 
    MarineWeatherOut, 
    PredictResponse, 
    HistoryResponse, 
    PredictionOut
)


# =====================================================================
# 4. SEED DATA (MASTER LOKASI NELAYAN JAWA TIMUR)
# =====================================================================
LOCATIONS_SEED = [
    {"name": "Pacitan",        "lat": -8.20, "lon": 111.10},
    {"name": "Prigi",          "lat": -8.29, "lon": 111.74},
    {"name": "Popoh",          "lat": -8.17, "lon": 111.89},
    {"name": "Sendang Biru",   "lat": -8.43, "lon": 112.69},
    {"name": "Puger",          "lat": -8.38, "lon": 113.47},
    {"name": "Pancer",         "lat": -8.63, "lon": 114.02},
    {"name": "Muncar",         "lat": -8.44, "lon": 114.33},
    {"name": "Grajagan",       "lat": -8.66, "lon": 114.23},
    {"name": "Watu Ulo",       "lat": -8.45, "lon": 113.72},
    {"name": "Blitar Selatan", "lat": -8.33, "lon": 112.19},
    {"name": "Tuban",          "lat": -6.90, "lon": 112.05},
    {"name": "Brondong",       "lat": -6.89, "lon": 112.27},
    {"name": "Paciran",        "lat": -6.87, "lon": 112.34},
    {"name": "Gresik",         "lat": -7.15, "lon": 112.65},
    {"name": "Surabaya",       "lat": -7.19, "lon": 112.65},
    {"name": "Pasuruan",       "lat": -7.64, "lon": 112.91},
    {"name": "Probolinggo",    "lat": -7.74, "lon": 113.23},
    {"name": "Situbondo",      "lat": -7.70, "lon": 114.00},
    {"name": "Banyuwangi",     "lat": -8.21, "lon": 114.37},
    {"name": "Bangkalan",      "lat": -6.95, "lon": 112.73},
    {"name": "Sampang",        "lat": -7.19, "lon": 113.24},
    {"name": "Pamekasan",      "lat": -7.16, "lon": 113.48},
    {"name": "Sumenep",        "lat": -7.02, "lon": 113.86},
    {"name": "Kangean",        "lat": -6.93, "lon": 115.32},
    {"name": "Selat Madura",   "lat": -7.10, "lon": 113.00},
    {"name": "Selat Bali",     "lat": -8.17, "lon": 114.43},
]


# =====================================================================
# 5. MANAJEMEN SIKLUS HIDUP (LIFESPAN)
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔴 KITA HAPUS load_all_models() DARI SINI AGAR VERCEL TIDAK CRASH SAAT BOOTING 🔴
    print("🚀 [SISTEM] Menginisialisasi sistem LENTERA LAUT di Vercel...")

    # Tahap 2: Inisialisasi Skema Basis Data
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Tahap 3: Penyemaian (Seeding) Data Lokasi Awal
    async with AsyncSessionLocal() as db:
        for loc in LOCATIONS_SEED:
            await get_or_create_location(db, loc["name"], loc["lat"], loc["lon"])

    yield


app = FastAPI(
    title="API LENTERA LAUT",
    description="Sistem Cerdas Prediksi Cuaca & Keselamatan Gelombang Nelayan Jawa Timur",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Mengizinkan akses dari web mana pun (sementara untuk testing)
    allow_credentials=True,
    allow_methods=["*"],  # Mengizinkan metode GET, POST, dll
    allow_headers=["*"],  # Mengizinkan semua header
)

# =====================================================================
# 6. ROUTER & ENDPOINTS
# =====================================================================

@app.get("/", tags=["Sistem Utama"])
def root():
    return {"message": "Sistem API Lentera Laut v2.0 beroperasi normal."}


# ---------------------------------------------------------------------
# GRUP 1: KATALOG LOKASI
# ---------------------------------------------------------------------
@app.get(
    "/locations",
    tags=["Katalog Lokasi"],
    response_model=list[LocationOut],
    summary="Menampilkan daftar 26 lokasi observasi nelayan"
)
async def list_locations(db: AsyncSession = Depends(get_db)):
    return await get_all_locations(db)


# ---------------------------------------------------------------------
# GRUP 2: OBSERVASI METEOROLOGI (RAW DATA)
# ---------------------------------------------------------------------
@app.get(
    "/marine-weather",
    tags=["Observasi Meteorologi"],
    response_model=list[MarineWeatherOut],
    summary="Mengambil histori cuaca mentah Open-Meteo per lokasi"
)
async def marine_weather(
    location: str = Query(..., description="Nama lokasi spesifik (contoh: Sendang Biru)"),
    limit: int    = Query(50, ge=1, le=500, description="Maksimal baris data"),
    db: AsyncSession = Depends(get_db)
):
    loc = await get_location_by_name(db, location)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Lokasi '{location}' tidak terdaftar.")
    
    return await get_marine_weather(db, loc.id_location, limit)


# ---------------------------------------------------------------------
# GRUP 3: SISTEM INFERENSI (MACHINE LEARNING) & RIWAYAT
# ---------------------------------------------------------------------
@app.get(
    "/predict",
    tags=["Sistem Inferensi (ML)"],
    response_model=PredictResponse,
    summary="Eksekusi prediksi cuaca & gelombang 1 jam ke depan"
)
async def predict(
    location: str = Query(..., description="Nama target lokasi prediksi"),
    db: AsyncSession = Depends(get_db)
):
    print(f"\n[ENGINE] 1. Validasi lokasi {location}...")
    loc = await get_location_by_name(db, location)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Lokasi '{location}' tidak ditemukan.")

    print("[ENGINE] 2. Menarik instrumen cuaca Open-Meteo API...")
    df = await run_in_threadpool(fetch_data, loc.latitude, loc.longitude)
    if df.empty:
        raise HTTPException(status_code=500, detail="Kegagalan penarikan data satelit Open-Meteo.")

    print("[ENGINE] 3. Sinkronisasi penyimpanan historis (PostgreSQL Upsert)...")
    await save_marine_weather(db, loc.id_location, df)

    print("[ENGINE] 4. Rekayasa Fitur Waktu (Feature Engineering)...")
    FEATURES = [
        "wave_height", "wind_speed_10m", "precipitation",            
        "visibility", "ocean_current_velocity", "sea_surface_temperature"    
    ]
    
    df_processed = df.sort_values("time").copy()
    for feat in FEATURES:
        df_processed[f"{feat}_lag1"] = df_processed[feat].shift(1)
        df_processed[f"{feat}_lag2"] = df_processed[feat].shift(2)
        df_processed[f"{feat}_lag3"] = df_processed[feat].shift(3)

    # Imputasi data aman sebelum diumpankan ke model
    df_processed = df_processed.ffill().bfill().fillna(0.0)
    last_row = df_processed.tail(1)
    
    lag_cols = [f"{feat}_lag{i}" for feat in FEATURES for i in (1, 2, 3)]
    X_live = last_row[lag_cols]
    
    print("[ENGINE] 5. Eksekusi Inferensi Ensemble ML (Main Thread)...")
    try:
        pred_dict = generate_forecast(X_live)
    except Exception as e:
        print(f"[ENGINE] ERROR MACHINE LEARNING: {e}")
        raise HTTPException(status_code=500, detail=f"Sistem ML gagal mengeksekusi matriks: {e}")

    print("[ENGINE] 6. Logging hasil prediksi ke basis data...")
    saved_pred = await save_prediction(db, loc.id_location, pred_dict)

    print("[ENGINE] 7. Selesai! Merakit JSON Response...\n")
    pred_dict["id_prediction"]   = saved_pred.id_prediction
    pred_dict["id_location"]     = loc.id_location
    pred_dict["time_prediction"] = saved_pred.time_prediction

    return {
        "location": {
            "id_location": loc.id_location,
            "name":        loc.name,
            "latitude":    loc.latitude,
            "longitude":   loc.longitude,
        },
        "prediction": pred_dict
    }

@app.get(
    "/history", 
    tags=["Sistem Inferensi (ML)"],
    response_model=HistoryResponse,
    summary="Menampilkan riwayat prediksi sebelumnya berdasarkan lokasi"
)
async def prediction_history(
    location: str = Query(..., description="Nama lokasi spesifik"),
    limit: int    = Query(10, ge=1, le=100, description="Jumlah riwayat terakhir"),
    db: AsyncSession = Depends(get_db)
):
    loc = await get_location_by_name(db, location)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Lokasi '{location}' tidak terdaftar.")

    history = await get_prediction_history(db, loc.id_location, limit)
    result = []
    
    for p in history:
        cat = p.categories
        result.append({
            "id_prediction":           p.id_prediction,
            "id_location":             p.id_location,
            "time_prediction":         p.time_prediction,
            "wave_height":             {"value": p.wave_height_pred,             "satuan": "meter",  "kategori": cat.wave_category if cat else None},
            "wind_speed_10m":          {"value": p.wind_speed_pred,              "satuan": "m/s",    "kategori": cat.wind_category if cat else None},
            "ocean_current_velocity":  {"value": p.ocean_current_velocity_pred,  "satuan": "m/s"},
            "sea_surface_temperature": {"value": p.sea_surface_temperature_pred, "satuan": "°C"},
            "precipitation":           {"value": p.precipitation_pred,           "satuan": "mm/jam", "kategori": cat.rain_category if cat else None},
            "visibility":              {"value": p.visibility_pred,              "satuan": "meter",  "kategori": cat.visibility_category if cat else None},
        })

    return {"location": loc.name, "history": result}

@app.get(
    "/prediction/{id_prediction}", 
    tags=["Sistem Inferensi (ML)"],
    response_model=PredictionOut,
    summary="Menarik spesifik 1 data prediksi berdasarkan ID"
)
async def prediction_detail(
    id_prediction: int,
    db: AsyncSession = Depends(get_db)
):
    pred = await get_prediction_by_id(db, id_prediction)
    if not pred:
        raise HTTPException(status_code=404, detail=f"Prediksi ID {id_prediction} tidak ditemukan pada sistem.")

    cat = pred.categories
    return {
        "id_prediction":             pred.id_prediction,
        "id_location":               pred.id_location,
        "time_prediction":           pred.time_prediction,
        "wave_height":               {"value": pred.wave_height_pred,             "satuan": "meter",  "kategori": cat.wave_category if cat else None},
        "wind_speed_10m":            {"value": pred.wind_speed_pred,              "satuan": "m/s",    "kategori": cat.wind_category if cat else None},
        "ocean_current_velocity":    {"value": pred.ocean_current_velocity_pred,  "satuan": "m/s"},
        "sea_surface_temperature":   {"value": pred.sea_surface_temperature_pred, "satuan": "°C"},
        "precipitation":             {"value": pred.precipitation_pred,           "satuan": "mm/jam", "kategori": cat.rain_category if cat else None},
        "visibility":                {"value": pred.visibility_pred,              "satuan": "meter",  "kategori": cat.visibility_category if cat else None},
    }


# ---------------------------------------------------------------------
# GRUP 4: PARTISIPASI NELAYAN (CROWDSOURCING & FEEDBACK)
# ---------------------------------------------------------------------
@app.post(
    "/users", 
    tags=["Partisipasi Nelayan"], 
    response_model=UserOut,
    summary="Mendaftarkan nelayan atau pengguna baru"
)
async def register_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    return await create_user(db, user)

@app.post(
    "/reports", 
    tags=["Partisipasi Nelayan"], 
    response_model=CatchReportOut,
    summary="Mengirim laporan hasil tangkapan dan validasi cuaca aktual"
)
async def submit_report(report: CatchReportCreate, db: AsyncSession = Depends(get_db)):
    # Validasi eksistensi lokasi sebelum menerima laporan
    # Jika perlu validasi lewat nama, gunakan get_location_by_name. Jika ID sudah cukup, bisa langsung di-pass.
    return await create_catch_report(db, report)

@app.get(
    "/reports/{id_location}", 
    tags=["Partisipasi Nelayan"], 
    response_model=list[CatchReportOut],
    summary="Melihat riwayat tangkapan di suatu lokasi spesifik"
)
async def view_reports(id_location: int, db: AsyncSession = Depends(get_db)):
    return await get_reports_by_location(db, id_location)