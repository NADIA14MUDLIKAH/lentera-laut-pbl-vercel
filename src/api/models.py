from sqlalchemy import Column, Integer, Float, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .database import Base

# =====================================================================
# 1. TABEL MASTER LOKASI (DIMENSI SPASIAL)
# =====================================================================
class Location(Base):
    __tablename__ = "locations"

    id_location = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name        = Column(String(100), nullable=False, unique=True)
    latitude    = Column(Float, nullable=False)
    longitude   = Column(Float, nullable=False)

    # Relasi: One-to-Many ke tabel anak
    marine_data   = relationship("MarineWeather", back_populates="location", cascade="all, delete-orphan")
    predictions   = relationship("Prediction", back_populates="location", cascade="all, delete-orphan")
    catch_reports = relationship("CatchReport", back_populates="location", cascade="all, delete-orphan")


# =====================================================================
# 2. TABEL DATA CUACA MENTAH (HISTORIS KRONOLOGIS)
# =====================================================================
class MarineWeather(Base):
    __tablename__ = "marine_weather"

    id_data                 = Column(Integer, primary_key=True, index=True, autoincrement=True)
    id_location             = Column(Integer, ForeignKey("locations.id_location", ondelete="CASCADE"), nullable=False)
    time                    = Column(DateTime(timezone=True), nullable=False)

    # Metrik Oseanografi & Meteorologi
    wave_height             = Column(Float, nullable=True)
    wind_speed_10m          = Column(Float, nullable=True)
    precipitation           = Column(Float, nullable=True)
    visibility              = Column(Float, nullable=True)
    ocean_current_velocity  = Column(Float, nullable=True)
    sea_surface_temperature = Column(Float, nullable=True)

    # Relasi Balik (Back-reference)
    location = relationship("Location", back_populates="marine_data")

    # Constraint Integritas: Mencegah duplikasi data pada titik lokasi dan waktu yang sama
    __table_args__ = (
        UniqueConstraint("id_location", "time", name="uq_location_time"),
    )


# =====================================================================
# 3. TABEL HASIL PREDIKSI (OUTPUT MACHINE LEARNING)
# =====================================================================
class Prediction(Base):
    __tablename__ = "predictions"

    id_prediction                = Column(Integer, primary_key=True, index=True, autoincrement=True)
    id_location                  = Column(Integer, ForeignKey("locations.id_location", ondelete="CASCADE"), nullable=False)
    time_prediction              = Column(DateTime(timezone=True), server_default=func.now())

    # Metrik Numerik Hasil Prediksi Model
    wave_height_pred             = Column(Float, nullable=True)
    wind_speed_pred              = Column(Float, nullable=True)
    precipitation_pred           = Column(Float, nullable=True)
    visibility_pred              = Column(Float, nullable=True)
    ocean_current_velocity_pred  = Column(Float, nullable=True)
    sea_surface_temperature_pred = Column(Float, nullable=True)

    # Relasi Hierarkis
    location   = relationship("Location", back_populates="predictions")
    # Parameter uselist=False memberlakukan kardinalitas One-to-One dengan tabel Category
    categories = relationship("Category", back_populates="prediction", cascade="all, delete-orphan", uselist=False)


# =====================================================================
# 4. TABEL KATEGORI (SISTEM PENDUKUNG KEPUTUSAN KESELAMATAN)
# =====================================================================
class Category(Base):
    __tablename__ = "categories"

    id_category         = Column(Integer, primary_key=True, index=True, autoincrement=True)
    id_prediction       = Column(Integer, ForeignKey("predictions.id_prediction", ondelete="CASCADE"), nullable=False, unique=True)

    # Label Klasifikasi Kualitatif
    wave_category       = Column(String(50), nullable=True) 
    wind_category       = Column(String(50), nullable=True) 
    rain_category       = Column(String(50), nullable=True) 
    visibility_category = Column(String(50), nullable=True) 

    # Relasi Balik
    prediction = relationship("Prediction", back_populates="categories")


# =====================================================================
# 5. TABEL PENGGUNA (NELAYAN / STAKEHOLDER)
# =====================================================================
class User(Base):
    __tablename__ = "users"

    id_user = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name    = Column(String(100), nullable=False)
    phone   = Column(String(20), unique=True, nullable=False) # Bisa untuk login via WA nanti
    origin  = Column(String(100), nullable=True) # Asal daerah nelayan

    # Relasi ke laporan tangkapan
    reports = relationship("CatchReport", back_populates="user", cascade="all, delete-orphan")


# =====================================================================
# 6. TABEL LAPORAN TANGKAPAN (CROWDSOURCING & MODEL FEEDBACK)
# =====================================================================
class CatchReport(Base):
    __tablename__ = "catch_reports"

    id_report         = Column(Integer, primary_key=True, index=True, autoincrement=True)
    id_user           = Column(Integer, ForeignKey("users.id_user", ondelete="CASCADE"), nullable=False)
    id_location       = Column(Integer, ForeignKey("locations.id_location", ondelete="CASCADE"), nullable=False)
    time_reported     = Column(DateTime(timezone=True), server_default=func.now())
    
    catch_weight_kg   = Column(Float, nullable=False)
    # Nelayan bisa memvalidasi prediksi kita: "Apakah badai?", "Gelombang tenang?", dsb.
    weather_condition = Column(String(50), nullable=True) 

    # Relasi Balik
    user     = relationship("User", back_populates="reports")
    location = relationship("Location", back_populates="catch_reports")