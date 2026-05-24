from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

# =====================================================================
# 1. SUB-SCHEMAS (KOMPONEN NESTED UNTUK JSON)
# =====================================================================
class MetricSimple(BaseModel):
    value: float
    satuan: str

class MetricWithCategory(BaseModel):
    value: float
    satuan: str
    kategori: Optional[str] = None


# =====================================================================
# 2. SCHEMAS: ENTITAS DATABASE (BASE MODELS)
# =====================================================================
class LocationOut(BaseModel):
    id_location: int
    name: str
    latitude: float
    longitude: float

    class Config:
        from_attributes = True


class MarineWeatherOut(BaseModel):
    id_data: int
    id_location: int
    time: datetime
    # Menambahkan = None agar aman saat menerima nilai Null dari database
    wave_height: Optional[float] = None
    wind_speed_10m: Optional[float] = None
    precipitation: Optional[float] = None
    visibility: Optional[float] = None
    ocean_current_velocity: Optional[float] = None
    sea_surface_temperature: Optional[float] = None

    class Config:
        from_attributes = True


# =====================================================================
# 3. SCHEMAS: HASIL PREDIKSI (OUTPUT PREDIKSI)
# =====================================================================
class PredictionOut(BaseModel):
    id_prediction: int
    id_location: int
    time_prediction: datetime
    
    # Atribut prediksi menggunakan sub-schema untuk validasi ketat
    wave_height: MetricWithCategory
    wind_speed_10m: MetricWithCategory
    ocean_current_velocity: MetricSimple
    sea_surface_temperature: MetricSimple
    precipitation: MetricWithCategory
    visibility: MetricWithCategory

    class Config:
        from_attributes = True


# =====================================================================
# 4. SCHEMAS: API RESPONSES (PAYLOAD AKHIR)
# =====================================================================
class PredictResponse(BaseModel):
    location: LocationOut
    prediction: PredictionOut


class HistoryResponse(BaseModel):
    location: str
    history: List[PredictionOut]

# =====================================================================
# 5. SCHEMAS: PENGGUNA & LAPORAN TANGKAPAN
# =====================================================================
class UserCreate(BaseModel):
    name: str
    phone: str
    origin: Optional[str] = None

class UserOut(BaseModel):
    id_user: int
    name: str
    phone: str
    origin: Optional[str]

    class Config:
        from_attributes = True

class CatchReportCreate(BaseModel):
    id_user: int
    id_location: int
    catch_weight_kg: float
    weather_condition: Optional[str] = None

class CatchReportOut(BaseModel):
    id_report: int
    id_user: int
    id_location: int
    time_reported: datetime
    catch_weight_kg: float
    weather_condition: Optional[str]

    class Config:
        from_attributes = True