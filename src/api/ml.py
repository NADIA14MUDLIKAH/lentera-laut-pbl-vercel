import os
import joblib
import numpy as np
import pandas as pd
import warnings

# =====================================================================
# 1. KONFIGURASI KEAMANAN INFRASTRUKTUR & PERINGATAN
# =====================================================================
# Menyembunyikan peringatan (warning) dari Scikit-Learn terkait hilangnya nama fitur
# saat DataFrame dikonversi menjadi array Numpy.
warnings.filterwarnings("ignore", category=UserWarning)

# [KRUSIAL] Variabel Lingkungan Anti-Deadlock untuk OS Windows
# Memaksa seluruh pustaka numerik pendukung untuk beroperasi dalam mode Single-Thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"


# =====================================================================
# 2. ATURAN AMBANG BATAS KATEGORISASI (DECISION RULES)
# =====================================================================
def kategori_gelombang(w: float) -> str:
    if w < 0.5:    return "Tenang"
    elif w < 1.25: return "Rendah"
    elif w < 2.5:  return "Sedang"
    elif w < 4.0:  return "Tinggi"
    elif w < 6.0:  return "Sangat Tinggi"
    else:          return "Ekstrem"

def kategori_angin(ws: float) -> str:
    if ws < 0.3:   return "Calm"
    elif ws < 1.6:  return "Light Air"
    elif ws < 3.4:  return "Light Breeze"
    elif ws < 5.5:  return "Gentle Breeze"
    elif ws < 8.0:  return "Moderate Breeze"
    elif ws < 10.8: return "Fresh Breeze"
    elif ws < 13.9: return "Strong Breeze"
    elif ws < 17.2: return "Near Gale"
    elif ws < 20.8: return "Gale"
    elif ws < 24.5: return "Strong Gale"
    elif ws < 28.5: return "Storm"
    elif ws < 32.7: return "Violent Storm"
    else:           return "Hurricane"

def kategori_hujan(p: float) -> str:
    if p < 0.1:    return "Tidak Hujan"
    elif p < 1.0:  return "Very Light"
    elif p < 5.0:  return "Light"
    elif p < 10.0: return "Moderate"
    elif p < 20.0: return "Heavy"
    else:          return "Extreme (Violent Rain)"

def kategori_visibility(v: float) -> str:
    score = min(100, (v / 24140) * 100)
    if score >= 90:   return "Excellent"
    elif score >= 70: return "Good"
    elif score >= 50: return "Fair"
    else:             return "Poor"


# =====================================================================
# 3. MANAJEMEN MODEL & LOGIKA INFERENSI
# =====================================================================
# Variabel global untuk menyimpan model di memori RAM agar API merespons dengan cepat
LOADED_MODELS = {}

TARGETS = [
    "wave_height", 
    "wind_speed_10m", 
    "ocean_current_velocity", 
    "sea_surface_temperature", 
    "precipitation", 
    "visibility"
]

def load_all_models():
    """
    Fungsi utilitas untuk memuat seluruh file serialisasi model (.pkl) ke dalam RAM.
    Menggunakan mekanisme caching agar tidak dibaca ulang setiap kali request masuk.
    """
    if LOADED_MODELS:
        return 

    # Rekonstruksi struktur direktori dinamis
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == "api":
        base_dir = os.path.dirname(current_dir)
    elif os.path.basename(current_dir) == "src":
        base_dir = os.path.dirname(current_dir)
    else:
        base_dir = current_dir

    models_dir = os.path.join(base_dir, "models", "saved_models")
    
    for target in TARGETS:
        # PERBAIKAN: Mengikuti format penamaan terbaru (model_[target].pkl)
        model_path = os.path.join(models_dir, f"model_{target}.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model tidak ditemukan di direktori absolut: {model_path}")
            
        # PERBAIKAN: Menggunakan joblib sesuai metode penyimpanan di skrip ekstraksi MLflow
        LOADED_MODELS[target] = joblib.load(model_path)


def generate_forecast(X_live: pd.DataFrame) -> dict:
    """
    Menerima matriks fitur lag (X_live), mengeksekusi inferensi pada 6 model ensemble 
    berbeda secara sekuensial, dan merakit respons JSON prediksi yang tervalidasi.
    """
    print("  [ML-ENGINE] Memeriksa status muatan model di memori...")
    load_all_models()
    
    results = {}
    
    print("  [ML-ENGINE] Mengonversi DataFrame menuju matriks Numpy murni...")
    X_live_np = X_live.to_numpy() 
    
    for target in TARGETS:
        print(f"  [ML-ENGINE] Melakukan inferensi untuk parameter: {target}...")
        
        model = LOADED_MODELS[target]
        
        # Override arsitektur internal model untuk mencegah multithreading deadlock
        try:
            # Atribut umum algoritma Scikit-Learn
            if hasattr(model, 'n_jobs'): model.n_jobs = 1
            if hasattr(model, 'nthread'): model.nthread = 1
            
            # Atribut spesifik modul XGBoost
            if hasattr(model, 'get_booster'):
                model.get_booster().set_param({'nthread': 1})
            
            # Atribut spesifik modul LightGBM
            elif hasattr(model, 'booster_'):
                model.booster_.params['num_threads'] = 1
        except Exception:
            pass # Lanjutkan proses jika model (misal: LinearRegression) tidak mendefinisikan batas thread
        
        # Eksekusi inferensi Machine Learning
        pred_val = float(model.predict(X_live_np)[0])
        
        # Penanganan nilai negatif yang secara fisis tidak logis
        if target in ["precipitation", "wave_height", "wind_speed_10m", "visibility"] and pred_val < 0:
            pred_val = 0.0
            
        results[target] = pred_val
        print(f"    -> [SUKSES] Prediksi {target}: {pred_val:.3f}")

    print("  [ML-ENGINE] Seluruh proses inferensi selesai secara sekuensial!")
    
    # Merakit objek respons yang sesuai dengan struktur Pydantic (schemas.py)
    return {
        "wave_height": {
            "value": round(results["wave_height"], 2),
            "satuan": "meter",
            "kategori": kategori_gelombang(results["wave_height"])
        },
        "wind_speed_10m": {
            "value": round(results["wind_speed_10m"], 2),
            "satuan": "m/s",
            "kategori": kategori_angin(results["wind_speed_10m"])
        },
        "ocean_current_velocity": {
            "value": round(results["ocean_current_velocity"], 2),
            "satuan": "m/s"
        },
        "sea_surface_temperature": {
            "value": round(results["sea_surface_temperature"], 2),
            "satuan": "°C"
        },
        "precipitation": {
            "value": round(results["precipitation"], 2),
            "satuan": "mm/jam",
            "kategori": kategori_hujan(results["precipitation"])
        },
        "visibility": {
            "value": round(results["visibility"], 0),
            "satuan": "meter",
            "kategori": kategori_visibility(results["visibility"])
        }
    }