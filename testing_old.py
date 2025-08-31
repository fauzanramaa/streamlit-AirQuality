import requests
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# API Key
API_KEY_OWM = "82b4a79b21a70c0bb31dc9915491b5c0"

# Load Models and Scalers
model_clf = joblib.load('models/model_aqi_classification_new.pkl')
model_lstm = load_model('models/model_lstm_forecasting.h5', compile=False)
model_lstm.compile(optimizer='adam', loss=MeanSquaredError())
scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')

aqi_classes = ['Baik', 'Sedang', 'Tidak Sehat untuk Kelompok Sensitif', 'Tidak Sehat', 'Sangat Tidak Sehat', 'Berbahaya']

def unix_to_local(unixtime):
    return datetime.utcfromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M')

def get_coordinates_from_city(city_name):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={city_name}"
    response = requests.get(url, headers={"User-Agent": "air-quality-app"})
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        return float(data['lat']), float(data['lon']), data['display_name']
    return None, None, "Tidak ditemukan"

def compute_iaqi(conc, bps):
    for c_low, c_high, i_low, i_high in bps:
        if c_low <= conc <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low
    return None

def compute_max_aqi(pm25, pm10, o3, no2, so2, co):
    bp_pm25 = [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
               (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
               (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
    bp_pm10 = [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
               (255, 354, 151, 200), (355, 424, 201, 300),
               (425, 504, 301, 400), (505, 604, 401, 500)]
    bp_o3 = [(0, 54, 0, 50), (55, 70, 51, 100), (71, 85, 101, 150),
             (86, 105, 151, 200), (106, 200, 201, 300)]
    bp_no2 = [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
              (361, 649, 151, 200), (650, 1249, 201, 300),
              (1250, 1649, 301, 400), (1650, 2049, 401, 500)]
    bp_so2 = [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
              (186, 304, 151, 200), (305, 604, 201, 300),
              (605, 804, 301, 400), (805, 1004, 401, 500)]
    bp_co = [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
             (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300),
             (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)]
    

    iaqi_vals = [
        compute_iaqi(pm25, bp_pm25),
        compute_iaqi(pm10, bp_pm10),
        compute_iaqi(o3, bp_o3),
        compute_iaqi(no2, bp_no2),
        compute_iaqi(so2, bp_so2),
        compute_iaqi(co, bp_co)
    ]
    iaqi_vals = [val for val in iaqi_vals if val is not None]
    return max(iaqi_vals) if iaqi_vals else None

def classification_aqi(pm25, pm10, o3, no2, so2, co):
    features = np.array([[pm25, pm10, o3, no2, so2, co]])
    index = model_clf.predict(features)[0]
    return aqi_classes[index]

def forecast_pm25_from_data(pm25_series, num_lags=4, rolling_means=(3, 4)):

    # Ambil lag terakhir sebanyak `num_lags` (dibalik agar urutan terbaru di depan)
    lags = list(pm25_series[-num_lags:][::-1])
    preds = []

    for i in range(4):
        # Hitung semua rolling mean yang diminta
        rolling_features = [np.mean(lags[:n]) for n in rolling_means]

        # Fitur lengkap: semua lag + rolling mean + weekday + is_weekend
        features = np.array([[
            *lags,
            *rolling_features,
            datetime.now().weekday(),
            1 if datetime.now().weekday() >= 5 else 0
        ]])

        # Transformasi dan prediksi
        scaled_input = scaler_X.transform(features)
        pred_scaled = model_lstm.predict(scaled_input.reshape(1, features.shape[1], 1))
        pred_value = scaler_y.inverse_transform(pred_scaled)[0][0]
        pred_value = np.clip(pred_value, 0, 500)

        # Simpan hasil prediksi dan update lags
        preds.append(pred_value)
        lags = [pred_value] + lags[:-1]  # geser lags

    return preds


# Program Utama

# Input Mode (1 = Manual, 2 = API)
while True:
    mode = input("\nPilih Mode Klasifikasi (1=Manual, 2=API): ").strip()
    if mode.isdigit() and int(mode) in (1, 2):
        mode = int(mode)
        break
    print("Input tidak valid! Harap masukkan angka 1 atau 2.")

# Input Nama Kota
while True:
    user_input = input("Masukkan nama kota: ").strip()
    if not user_input:
        print("Input tidak boleh kosong.")
    elif user_input.replace(' ', '').isalpha():
        city_name = user_input.lower()
        break
    else:
        print("Input tidak valid! Hanya huruf dan spasi yang diperbolehkan.")

# Ambil Koordinat Kota
LAT, LON, LOC_NAME = get_coordinates_from_city(city_name)
if LAT is None:
    print(f"❌ Gagal mendapatkan koordinat kota '{city_name}': {LOC_NAME}")
    exit()

# --- MODE API ---
if mode == 2:
    api_url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={LAT}&lon={LON}&appid={API_KEY_OWM}"
    )
    res = requests.get(api_url)

    if res.status_code == 200:
        comp = res.json()['list'][0]['components']
        pm25, pm10, o3, no2, so2, co = (
            comp['pm2_5'], comp['pm10'], comp['o3'],
            comp['no2'], comp['so2'], comp['co']
        )

        print("\n🌐 Real-time Data Polusi Udara (API):")
        for key in ['pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co']:
            print(f"{key.upper():<6} = {comp[key]:.2f} µg/m³")
    else:
        print("❌ Gagal mengambil data polusi dari API.")
        exit()

# --- MODE MANUAL ---
else:
    try:
        pm25 = float(input("PM2.5: ") or 0)
        pm10 = float(input("PM10: ") or 0)
        o3   = float(input("O3: ") or 0)
        no2  = float(input("NO2: ") or 0)
        so2  = float(input("SO2: ") or 0)
        co   = float(input("CO: ") or 0)
    except ValueError:
        print("❌ Input tidak valid. Harap masukkan angka.")
        exit()

# --- Proses Klasifikasi dan Hasil ---
predicted_class = classification_aqi(pm25, pm10, o3, no2, so2, co)
max_aqi = compute_max_aqi(pm25, pm10, o3, no2, so2, co)

print(f"\n✅ Klasifikasi Udara untuk {LOC_NAME}")
print(f"AQI Maksimum : {round(max_aqi)}")
print(f"Kategori     : {predicted_class}")


# ===============================
# 1️⃣ Ambil Data Historis (4 Hari Terakhir)
# ===============================
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
        pm25 = item["components"].get("pm2_5")
        dt_obj = datetime.utcfromtimestamp(item["dt"])
        historical_records.append({
            "date": dt_obj.date(),
            "pm2_5": pm25
        })
else:
    print("❌ Gagal mengambil data historis")
    exit()

# ===========================
# 2️⃣ Proses dan Tampilkan Data Historis
# ===========================
if historical_records:
    df_history = pd.DataFrame(historical_records)
    # Filter ulang untuk memastikan hanya tanggal 4 hari lalu sampai kemarin
    start_date = (datetime.utcnow().date() - timedelta(days=4))
    end_date = (datetime.utcnow().date() - timedelta(days=1))
    df_history = df_history[(df_history['date'] >= start_date) & (df_history['date'] <= end_date)]
    
    df_history = df_history.groupby("date").mean(numeric_only=True).reset_index()
    df_history = df_history.sort_values("date").reset_index(drop=True)

    print("\n✅ Historical data (PM2.5 - µg/m³):")
    for _, row in df_history.iterrows():
        print(f"{row['date']} - Avg PM2.5: {row['pm2_5']:.2f} µg/m³")

    historical_dates = set(df_history['date'])

    # ========================
    # 3️⃣ Prediksi dengan Model Sendiri (LSTM)
    # ========================
    predicted_pm25 = forecast_pm25_from_data(df_history["pm2_5"].tolist())
    print("\n📈 Prediksi PM2.5 4 Hari ke Depan (Model LSTM):")
    start_date = datetime.utcnow().date() + timedelta(days=1)
    for i, val in enumerate(predicted_pm25):
        pred_date = start_date + timedelta(days=i)
        print(f"{pred_date} - Prediksi PM2.5: {val:.2f} µg/m³")

else:
    print("❌ Tidak ada data historis yang berhasil diambil.")
    exit()

# # ============================
# # 4️⃣ Forecast Data dari OpenWeatherMap API
# # ============================
# url_forecast = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={API_KEY_OWM}"
# response_forecast = requests.get(url_forecast)

# if response_forecast.status_code == 200:
#     data = response_forecast.json()
#     forecast_list = data["list"]

#     forecast_records = []
#     for item in forecast_list:
#         dt = datetime.utcfromtimestamp(item["dt"])
#         pm25 = item["components"].get("pm2_5", None)
#         forecast_records.append({"date": dt.date(), "pm2_5": pm25})

#     df_forecast = pd.DataFrame(forecast_records)
#     df_forecast = df_forecast.groupby("date").mean(numeric_only=True).reset_index()

#     # Buang tanggal historis dan hari ini
#     today = datetime.utcnow().date()
#     df_forecast = df_forecast[~df_forecast["date"].isin(historical_dates.union({today}))]
#     df_forecast = df_forecast.sort_values("date")

#     if not df_forecast.empty:
#         print("\n📊 Data Aktual PM2.5 dari API (4 Hari ke Depan):")
#         for _, row in df_forecast.iterrows():
#             print(f"{row['date']} - Avg PM2.5: {row['pm2_5']:.2f} µg/m³")
#     else:
#         print("❌ Tidak ada data forecast yang bisa ditampilkan.")
# else:
#     print("❌ Gagal mengambil data forecast OWM:", response_forecast.text)



# # ============================
# # 5️⃣ Ringkasan Akurasi Prediksi
# # ============================

# actuals = []
# preds = []

# for i in range(min(4, len(predicted_pm25), len(df_forecast))):
#     actual = df_forecast.iloc[i]['pm2_5']
#     predicted = predicted_pm25[i]

#     if pd.notna(actual) and actual != 0:
#         actuals.append(actual)
#         preds.append(predicted)

# if actuals and preds:
#     # Konversi ke array numpy
#     actuals = np.array(actuals)
#     preds = np.array(preds)

#     # MAE & RMSE
#     mae = mean_absolute_error(actuals, preds)
#     rmse = mean_squared_error(actuals, preds, squared=False)

#     # R²
#     r2 = r2_score(actuals, preds)

#     # Akurasi sederhana berbasis MAE
#     avg_actual = np.mean(actuals)
#     accuracy = max(0, 100 - (mae / avg_actual * 100))

#     print("\n📊 Ringkasan Akurasi Prediksi vs Data API:")
#     print(f"   🔹 MAE  : {mae:.2f} µg/m³")
#     print(f"   🔹 RMSE : {rmse:.2f} µg/m³")
#     print(f"   🔹 R²   : {r2:.4f}")
#     print(f"   🎯 Akurasi Prediksi: {accuracy:.2f}%")
# else:
#     print("❌ Data untuk evaluasi akurasi tidak cukup.")





