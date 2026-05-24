import os
import requests
import pandas as pd

def main():
    # Definisikan path tujuan (berdasarkan root direktori proyek)
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    locations = [
        {"name": "Pacitan", "lat": -8.20, "lon": 111.10},
        {"name": "Prigi", "lat": -8.29, "lon": 111.74},
        {"name": "Popoh", "lat": -8.17, "lon": 111.89},
        {"name": "Sendang Biru", "lat": -8.43, "lon": 112.69},
        {"name": "Puger", "lat": -8.38, "lon": 113.47},
        {"name": "Pancer", "lat": -8.63, "lon": 114.02},
        {"name": "Muncar", "lat": -8.44, "lon": 114.33},
        {"name": "Grajagan", "lat": -8.66, "lon": 114.23},
        {"name": "Watu Ulo", "lat": -8.45, "lon": 113.72},
        {"name": "Blitar Selatan", "lat": -8.33, "lon": 112.19},
        {"name": "Tuban", "lat": -6.90, "lon": 112.05},
        {"name": "Brondong", "lat": -6.89, "lon": 112.27},
        {"name": "Paciran", "lat": -6.87, "lon": 112.34},
        {"name": "Gresik", "lat": -7.15, "lon": 112.65},
        {"name": "Surabaya", "lat": -7.19, "lon": 112.65},
        {"name": "Pasuruan", "lat": -7.64, "lon": 112.91},
        {"name": "Probolinggo", "lat": -7.74, "lon": 113.23},
        {"name": "Situbondo", "lat": -7.70, "lon": 114.00},
        {"name": "Banyuwangi", "lat": -8.21, "lon": 114.37},
        {"name": "Bangkalan", "lat": -6.95, "lon": 112.73},
        {"name": "Sampang", "lat": -7.19, "lon": 113.24},
        {"name": "Pamekasan", "lat": -7.16, "lon": 113.48},
        {"name": "Sumenep", "lat": -7.02, "lon": 113.86},
        {"name": "Kangean", "lat": -6.93, "lon": 115.32},
        {"name": "Selat Madura", "lat": -7.10, "lon": 113.00},
        {"name": "Selat Bali", "lat": -8.17, "lon": 114.43},
    ]

    latitudes = [loc["lat"] for loc in locations]
    longitudes = [loc["lon"] for loc in locations]
    location_names = [loc["name"] for loc in locations]

    print("⚡ Menarik data batch dari Open-Meteo API...")

    try:
        # 1. Panggil Marine API
        marine_url = "https://marine-api.open-meteo.com/v1/marine"
        marine_params = {
            "latitude": latitudes, "longitude": longitudes,
            "hourly": ["wave_height", "ocean_current_velocity", "sea_surface_temperature"],
            "timezone": "Asia/Jakarta"
        }
        marine_data = requests.get(marine_url, params=marine_params, timeout=30).json()

        # 2. Panggil Weather API
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitudes, "longitude": longitudes,
            "hourly": ["wind_speed_10m", "precipitation", "visibility"],
            "timezone": "Asia/Jakarta"
        }
        weather_data = requests.get(weather_url, params=weather_params, timeout=30).json()

    except Exception as e:
        print(f"❌ Kkoneksi API gagal: {e}")
        return

    if not isinstance(marine_data, list):
        marine_data = [marine_data]
        weather_data = [weather_data]

    all_dfs = []

    # 3. Ekstrak, Urutkan Kolom, dan Ekspor Parsial
    for i, name in enumerate(location_names):
        m_df = pd.DataFrame(marine_data[i]["hourly"])
        w_df = pd.DataFrame(weather_data[i]["hourly"])
        
        loc_df = pd.merge(m_df, w_df, on="time")
        loc_df["location"] = name
        
        # Format susunan kolom sesuai permintaan
        ordered_columns = [
            "time", "location", "wave_height", "wind_speed_10m", 
            "precipitation", "visibility", "ocean_current_velocity", "sea_surface_temperature"
        ]
        loc_df = loc_df[ordered_columns]
        
        # Simpan file parsial (contoh: data/raw/pacitan.csv)
        clean_name = name.lower().replace(" ", "_")
        loc_df.to_csv(os.path.join(raw_dir, f"{clean_name}.csv"), index=False)
        all_dfs.append(loc_df)

    # 4. Gabungkan seluruh lokasi ke berkas master
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_df.to_csv(os.path.join(raw_dir, "all_locations.csv"), index=False)
    print(f"✅ Target Selesai: all_locations.csv berhasil dibuat dengan {len(master_df)} baris.")

if __name__ == "__main__":
    main()