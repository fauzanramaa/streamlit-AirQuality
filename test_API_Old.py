import requests
import json
from datetime import datetime, timedelta
from collections import defaultdict
import os

# 🔑 API Key dan lokasi
API_KEY_OWM = "82b4a79b21a70c0bb31dc9915491b5c0"
lat = -6.2  # Jakarta
lon = 106.8

def unix_to_str(ts):
    dt = datetime.utcfromtimestamp(ts)
    return dt.strftime('%Y-%m-%d %H:%M (%A)')

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"💾 JSON disimpan di: {filename}")

# ================================
# 1️⃣ Real-time Air Pollution Data
# ================================
print("===== 🔵 REAL-TIME DATA =====")
url_realtime = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY_OWM}"
res_realtime = requests.get(url_realtime)
data_rt = res_realtime.json()

for i, item in enumerate(data_rt.get("list", [])):
    dt_str = unix_to_str(item["dt"])
    print(f"\n🕒 Waktu: {dt_str}")
    print("📦 Komponen:")
    print(json.dumps(item["components"], indent=2))

# ================================
# 2️⃣ Forecast Air Pollution Data
# ================================
print("\n\n===== 🟢 FORECASTING DATA =====")
url_forecast = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={lat}&lon={lon}&appid={API_KEY_OWM}"
res_forecast = requests.get(url_forecast)
data_fc = res_forecast.json()
list_fc = data_fc.get("list", [])

# ➕ Tampilkan 3 entri pertama
print("\n🧾 3 Entri Pertama:")
for i, item in enumerate(list_fc[:3]):
    dt_str = unix_to_str(item["dt"])
    print(f"\n⏱️ Waktu: {dt_str}")
    print(json.dumps(item, indent=2))

# ➕ Hitung jumlah entri
print(f"\n📊 Jumlah entri forecasting (jam): {len(list_fc)}")

# ➕ Kelompokkan berdasarkan hari
grouped_fc = defaultdict(list)
for item in list_fc:
    date = datetime.utcfromtimestamp(item["dt"]).strftime('%Y-%m-%d')
    grouped_fc[date].append(item)

print(f"📆 Jumlah hari unik forecasting: {len(grouped_fc)} hari")

# ➕ Hitung rata-rata PM2.5 per hari
print("\n📈 Rata-rata PM2.5 per Hari (Forecast):")
for date, items in grouped_fc.items():
    pm25_avg = sum(i['components']['pm2_5'] for i in items) / len(items)
    print(f"{date} → {pm25_avg:.2f} µg/m³")

# ➕ Simpan JSON mentah
save_json(data_fc, "forecast_raw.json")


# ================================
# 3️⃣ Historical Air Pollution Data
# ================================
print("\n\n===== 🔴 HISTORICAL DATA (MAKSIMUM 5 HARI) =====")
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=5)

start_ts = int(start_time.timestamp())
end_ts = int(end_time.timestamp())

url_history = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={API_KEY_OWM}"
res_history = requests.get(url_history)
data_hist = res_history.json()
list_hist = data_hist.get("list", [])

# ➕ Tampilkan 3 entri pertama
print("\n🧾 3 Entri Pertama:")
for i, item in enumerate(list_hist[:3]):
    dt_str = unix_to_str(item["dt"])
    print(f"\n⏱️ Waktu: {dt_str}")
    print(json.dumps(item, indent=2))

# ➕ Hitung jumlah entri
print(f"\n📊 Jumlah entri historis (jam): {len(list_hist)}")

# ➕ Kelompokkan berdasarkan hari
grouped_hist = defaultdict(list)
for item in list_hist:
    date = datetime.utcfromtimestamp(item["dt"]).strftime('%Y-%m-%d')
    grouped_hist[date].append(item)

print(f"📆 Jumlah hari unik historis: {len(grouped_hist)} hari")

# ➕ Hitung rata-rata PM2.5 per hari
print("\n📈 Rata-rata PM2.5 per Hari (Historis):")
for date, items in grouped_hist.items():
    pm25_avg = sum(i['components']['pm2_5'] for i in items) / len(items)
    print(f"{date} → {pm25_avg:.2f} µg/m³")

# ➕ Simpan JSON mentah
save_json(data_hist, "historical_raw.json")
