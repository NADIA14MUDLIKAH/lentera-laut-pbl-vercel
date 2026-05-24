import pandas as pd
import requests
import numpy as np
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload
from sqlalchemy.dialects.postgresql import insert

# Pastikan class ini diimpor persis sesuai nama di models.py
from .models import Location, MarineWeather, Prediction, Category
from .models import User, CatchReport
from .schemas import UserCreate, CatchReportCreate

# =====================================================================
# 1. FETCH DATA OPEN-METEO
# =====================================================================
def fetch_data(lat: float, lon: float) -> pd.DataFrame:
    try:
        marine = requests.get(
            "https://marine-api.open-meteo.com/v1/marine",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "wave_height,ocean_current_velocity,sea_surface_temperature",
                "timezone": "Asia/Jakarta"
            },
            timeout=10
        ).json()

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "wind_speed_10m,precipitation,visibility",
                "timezone": "Asia/Jakarta"
            },
            timeout=10
        ).json()

        # SAFE GUARD: Validasi respons API
        if "hourly" not in marine or "hourly" not in weather:
            raise ValueError("API response tidak valid dari Open-Meteo")

        df_marine = pd.DataFrame(marine["hourly"])
        df_weather = pd.DataFrame(weather["hourly"])

        # Menggabungkan data cuaca maritim dan atmosfer
        df = pd.merge(df_marine, df_weather, on="time", how="inner")
        df["time"] = pd.to_datetime(df["time"])

        # Isolasi kolom opsional jika API sedang bermasalah/kosong
        df["visibility"] = df.get("visibility", 10000)

        # Ubah nilai NaN/NaT Pandas menjadi None agar PostgreSQL bisa memahaminya sebagai NULL
        df = df.replace({np.nan: None})

        return df

    except Exception as e:
        print(f"[ERROR] FETCH DATA API: {e}")
        return pd.DataFrame()


# =====================================================================
# 2. CRUD: LOKASI NELAYAN
# =====================================================================
async def get_or_create_location(db: AsyncSession, name: str, lat: float, lon: float):
    result = await db.execute(select(Location).where(Location.name == name))
    loc = result.scalar_one_or_none()

    if not loc:
        loc = Location(name=name, latitude=lat, longitude=lon)
        db.add(loc)
        await db.commit()
        await db.refresh(loc)

    return loc

async def get_all_locations(db: AsyncSession):
    result = await db.execute(select(Location).order_by(Location.name))
    return result.scalars().all()

async def get_location_by_name(db: AsyncSession, name: str):
    result = await db.execute(select(Location).where(Location.name == name))
    return result.scalar_one_or_none()


# =====================================================================
# 3. CRUD: DATA CUACA MENTAH (MARINE WEATHER)
# =====================================================================
async def save_marine_weather(db: AsyncSession, id_location: int, df: pd.DataFrame):
    if df.empty:
        return

    # Siapkan semua data dari DataFrame menjadi format dictionary batch
    records = []
    for _, row in df.iterrows():
        records.append({
            "id_location": id_location,
            "time": row["time"].to_pydatetime(),
            "wave_height": row.get("wave_height"),
            "wind_speed_10m": row.get("wind_speed_10m"),
            "precipitation": row.get("precipitation"),
            "visibility": row.get("visibility"),
            "ocean_current_velocity": row.get("ocean_current_velocity"),
            "sea_surface_temperature": row.get("sea_surface_temperature"),
        })

    # Menggunakan Postgres UPSERT (INSERT ... ON CONFLICT DO NOTHING)
    stmt = insert(MarineWeather).values(records)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=['id_location', 'time']
    )

    await db.execute(stmt)
    await db.commit()

async def get_marine_weather(db: AsyncSession, id_location: int, limit: int = 50):
    result = await db.execute(
        select(MarineWeather)
        .where(MarineWeather.id_location == id_location)
        .order_by(desc(MarineWeather.time))
        .limit(limit)
    )
    return result.scalars().all()


# =====================================================================
# 4. CRUD: HASIL PREDIKSI MACHINE LEARNING
# =====================================================================
async def save_prediction(db: AsyncSession, id_location: int, pred_results: dict):
    
    # 🔴 WAKTU MASA DEPAN (1 Jam setelah eksekusi API)
    waktu_masa_depan = datetime.now() + timedelta(hours=1)

    prediction = Prediction(
        id_location=id_location,
        time_prediction=waktu_masa_depan, 
        wave_height_pred=pred_results["wave_height"]["value"],
        wind_speed_pred=pred_results["wind_speed_10m"]["value"],
        ocean_current_velocity_pred=pred_results["ocean_current_velocity"]["value"],
        sea_surface_temperature_pred=pred_results["sea_surface_temperature"]["value"],
        precipitation_pred=pred_results["precipitation"]["value"],
        visibility_pred=pred_results["visibility"]["value"],
    )

    db.add(prediction)
    # Flush agar ID Prediksi terbentuk dan bisa dipakai untuk relasi tabel Kategori
    await db.flush() 

    category = Category(
        id_prediction=prediction.id_prediction,
        wave_category=pred_results["wave_height"].get("kategori"),
        wind_category=pred_results["wind_speed_10m"].get("kategori"),
        rain_category=pred_results["precipitation"].get("kategori"),
        visibility_category=pred_results["visibility"].get("kategori"),
    )

    db.add(category)
    await db.commit()
    await db.refresh(prediction)

    return prediction

async def get_prediction_history(db: AsyncSession, id_location: int, limit: int = 10):
    result = await db.execute(
        select(Prediction)
        .where(Prediction.id_location == id_location)
        .order_by(desc(Prediction.time_prediction))
        # Mengambil data relasional di tabel category sekaligus (Eager Loading)
        .options(selectinload(Prediction.categories)) 
        .limit(limit)
    )
    return result.scalars().all()

async def get_prediction_by_id(db: AsyncSession, id_prediction: int):
    result = await db.execute(
        select(Prediction)
        .where(Prediction.id_prediction == id_prediction)
        .options(selectinload(Prediction.categories))
    )
    return result.scalar_one_or_none()

# =====================================================================
# 5. CRUD: USERS & CATCH REPORTS
# =====================================================================
async def create_user(db: AsyncSession, user_in: UserCreate):
    # Cek apakah nomor HP sudah terdaftar
    result = await db.execute(select(User).where(User.phone == user_in.phone))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        return existing_user # Kembalikan user lama jika sudah ada

    new_user = User(name=user_in.name, phone=user_in.phone, origin=user_in.origin)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

async def create_catch_report(db: AsyncSession, report_in: CatchReportCreate):
    new_report = CatchReport(
        id_user=report_in.id_user,
        id_location=report_in.id_location,
        catch_weight_kg=report_in.catch_weight_kg,
        weather_condition=report_in.weather_condition
    )
    db.add(new_report)
    await db.commit()
    await db.refresh(new_report)
    return new_report

async def get_reports_by_location(db: AsyncSession, id_location: int, limit: int = 20):
    result = await db.execute(
        select(CatchReport)
        .where(CatchReport.id_location == id_location)
        .order_by(desc(CatchReport.time_reported))
        .limit(limit)
    )
    return result.scalars().all()