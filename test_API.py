import requests
import pandas as pd
from datetime import datetime, timedelta

# 🔑 API Key
API_KEY_OWM = "82b4a79b21a70c0bb31dc9915491b5c0"
API_KEY_IQAIR = "a3873792-963e-4c84-8ce2-c74ffef325e0"

# ======================================
# 🔧 Fungsi bantu
# ======================================

def unix_to_local(unixtime): 
    """Konversi UNIX timestamp ke waktu lokal."""
    return datetime.utcfromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M')

def get_coordinates_from_city(city_name):
    """Ambil koordinat dan nama lengkap lokasi dari nama kota."""
    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={city_name}"
        response = requests.get(url, headers={"User-Agent": "air-quality-app"})
        if response.status_code == 200:
            data = response.json()
            if data:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                display_name = data[0]['display_name']
                return lat, lon, display_name
            else:
                return None, None, "Tidak ditemukan"
        else:
            return None, None, f"Gagal: {response.status_code}"
    except Exception as e:
        return None, None, f"Error: {e}"

# ======================================
# 1️⃣ INPUT: Nama kota
# ======================================

city_name = input("Masukkan nama kota: ").strip().lower()
LAT, LON, location_name = get_coordinates_from_city(city_name)

if LAT is not None and LON is not None:

    # ======================================
    # 2️⃣ Real-Time Data - OpenWeatherMap
    # ======================================
    url_realtime = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY_OWM}"
    response_realtime = requests.get(url_realtime)

    if response_realtime.status_code == 200:
        data = response_realtime.json()
        components = data["list"][0]["components"]
        print("\n✅ Real-time data (OpenWeatherMap):")
        print(f"Lokasi: {location_name}")
        print(f"Waktu: {unix_to_local(data['list'][0]['dt'])}")
        for pol, value in components.items():
            print(f"{pol.upper()} = {value:.2f} µg/m³")
    else:
        print("❌ Gagal mengambil data real-time OWM:", response_realtime.text)

    # ======================================
    # 3️⃣ Historical Data - 4 hari ke belakang (tanpa hari ini)
    # ======================================
    historical_records = []

    # Hitung waktu start (4 hari lalu jam 00:00 UTC)
    start_dt = datetime.utcnow() - timedelta(days=4)
    start_ts = int(start_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())

    # Hitung waktu end (hari kemarin jam 23:59:59 UTC)
    end_dt = datetime.utcnow() - timedelta(days=1)
    end_ts = int(end_dt.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp())

    url_history = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={API_KEY_OWM}"

    response = requests.get(url_history)

    if response.status_code == 200:
        data = response.json()
        for item in data["list"]:
            pm25 = item["components"].get("pm2_5", None)
            dt_obj = datetime.utcfromtimestamp(item["dt"])
            historical_records.append({"date": dt_obj.date(), "pm2_5": pm25})
    else:
        print(f"❌ Gagal mengambil data historis")

    if historical_records:
        df_history = pd.DataFrame(historical_records)
        df_history = df_history[df_history['date'] >= (datetime.utcnow().date() - timedelta(days=4))]
        df_history = df_history[df_history['date'] <= (datetime.utcnow().date() - timedelta(days=1))]
        df_history = df_history.groupby("date").mean(numeric_only=True).reset_index()
        historical_dates = set(df_history["date"])

        print("\n✅ Historical data (PM2.5 - µg/m³):")
        for _, row in df_history.iterrows():
            print(f"{row['date']} - Avg PM2.5: {row['pm2_5']:.2f} µg/m³")
    else:
        print("❌ Tidak ada data historis yang berhasil diambil.")
        historical_dates = set()


    # ======================================
    # 4️⃣ Forecast Data - OpenWeatherMap
    # ======================================
    url_forecast = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={API_KEY_OWM}"
    response_forecast = requests.get(url_forecast)

    if response_forecast.status_code == 200:
        data = response_forecast.json()
        forecast_list = data["list"]

        forecast_records = []
        for item in forecast_list:
            dt = datetime.utcfromtimestamp(item["dt"])
            pm25 = item["components"].get("pm2_5", None)
            forecast_records.append({"date": dt.date(), "pm2_5": pm25})

        df_forecast = pd.DataFrame(forecast_records)
        df_forecast = df_forecast.groupby("date").mean(numeric_only=True).reset_index()

        # Exclude tanggal yang sudah tampil di historical dan hari ini
        today = datetime.utcnow().date()
        df_forecast = df_forecast[~df_forecast["date"].isin(historical_dates.union({today}))]

        print("\n✅ Forecast harian (PM2.5 - µg/m³):")
        for _, row in df_forecast.iterrows():
            print(f"{row['date']} - Avg PM2.5: {row['pm2_5']:.2f} µg/m³")
    else:
        print("❌ Gagal mengambil data forecast OWM:", response_forecast.text)

    # ======================================
    # 5️⃣ Real-Time Data - IQAir
    # ======================================
    url_iqair = f"https://api.airvisual.com/v2/nearest_city?lat={LAT}&lon={LON}&key={API_KEY_IQAIR}"
    response_iqair = requests.get(url_iqair)

    if response_iqair.status_code == 200:
        data = response_iqair.json()
        city = data['data']['city']
        country = data['data']['country']
        datetime_obs = data['data']['current']['pollution']['ts']
        aqi_us = data['data']['current']['pollution']['aqius']
        main_pollutant = data['data']['current']['pollution']['mainus']

        temp = data['data']['current']['weather']['tp']
        humidity = data['data']['current']['weather']['hu']
        pressure = data['data']['current']['weather']['pr']

        print("\n✅ Real-time data (IQAir):")
        print(f"Lokasi        : {city}, {country}")
        print(f"Waktu         : {datetime_obs}")
        print(f"AQI (US)      : {aqi_us} (skala AQI-US)")
        print(f"Polutan utama : {main_pollutant}")
        print(f"Suhu          : {temp}°C")
        print(f"Kelembapan    : {humidity}%")
        print(f"Tekanan       : {pressure} hPa")
    else:
        print("❌ Gagal mengambil data dari IQAir:", response_iqair.text)

else:
    print(f"❌ Gagal menemukan lokasi '{city_name}': {location_name}")
