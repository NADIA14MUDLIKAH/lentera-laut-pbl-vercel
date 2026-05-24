import pandas as pd

def clean_and_build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Fungsi untuk menangani missing value dengan ffill & bfill,
    lalu membuat fitur lag (1, 2, 3) untuk kebutuhan pemodelan.
    """
    # Copy dataframe agar tidak mengubah data asli
    df = df_raw.copy()
    
    # 1. Handling Missing Value (Imputasi Kontinuitas Waktu)
    df = df.ffill()
    df = df.bfill()
    
    # 2. Time Conversion & Sorting
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(by=["location", "time"]).reset_index(drop=True)
    
    # 3. Feature Engineering (Pembuatan Jendela Riwayat / Lag Features)
    features_to_lag = [
        "wave_height", "wind_speed_10m", "precipitation", 
        "visibility", "ocean_current_velocity", "sea_surface_temperature"
    ]

    for col in features_to_lag:
        df[f"{col}_lag1"] = df.groupby("location")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("location")[col].shift(2)
        df[f"{col}_lag3"] = df.groupby("location")[col].shift(3)
        
    # Hapus baris awal yang bernilai NaN akibat pergeseran lag
    df_clean = df.dropna().reset_index(drop=True)
    
    return df_clean