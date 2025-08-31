# app.py (Premium UI)
import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
import json
import os
import base64
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
import plotly.graph_objects as go
import plotly.express as px

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Air Quality Monitor", layout="wide", initial_sidebar_state="auto")
OPENWEATHER_API_KEY = "82b4a79b21a70c0bb31dc9915491b5c0"  # Ganti kalau perlu
MODEL_DIR = "models"
ALERT_AUDIO_PATH = "alert.mp3"  # optional

# -------------------------
# CSS (card style + minor theming)
# -------------------------
st.markdown(
    """
    <style>
    /* Card */
    .card {
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
      background: var(--container-bg);
      margin-bottom: 12px;
    }
    .muted { color: var(--secondary-text); font-size:14px; }
    .big { font-size:28px; font-weight:700; }
    .small { font-size:12px; color:var(--secondary-text); }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Helpers & model loading
# -------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        clf = joblib.load(os.path.join(MODEL_DIR, "model_aqi_classification_new.pkl"))
        lstm = load_model(os.path.join(MODEL_DIR, "model_lstm_forecasting.h5"), compile=False)
        lstm.compile(optimizer='adam', loss=MeanSquaredError())
        scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
        scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))
        return clf, lstm, scaler_X, scaler_y
    except Exception as e:
        return None, None, None, None

def get_coords(city):
    url = f"https://nominatim.openstreetmap.org/search?format=json&q={city}"
    try:
        r = requests.get(url, headers={"User-Agent":"air-quality-app"}, timeout=8)
        if r.status_code == 200 and r.json():
            d = r.json()[0]
            return float(d["lat"]), float(d["lon"]), d.get("display_name", city)
    except:
        pass
    return None, None, None

def fetch_owm_realtime(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        comp = r.json()["list"][0]["components"]
        return {
            "pm25": float(comp.get("pm2_5", 0)),
            "pm10": float(comp.get("pm10", 0)),
            "o3": float(comp.get("o3", 0)),
            "no2": float(comp.get("no2", 0)),
            "so2": float(comp.get("so2", 0)),
            "co": float(comp.get("co", 0))
        }
    raise RuntimeError("Failed to fetch OWM realtime")

def fetch_owm_history(lat, lon, days_back=4):
    start_dt = datetime.utcnow() - timedelta(days=days_back)
    start_ts = int(start_dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    end_dt = datetime.utcnow() - timedelta(days=1)
    end_ts = int(end_dt.replace(hour=23, minute=59, second=59, microsecond=0).timestamp())
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={OPENWEATHER_API_KEY}"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        items = r.json().get("list", [])
        hist = []
        for it in items:
            dt = datetime.utcfromtimestamp(it["dt"]).date()
            hist.append({"date": dt, "pm2_5": it["components"].get("pm2_5", None)})
        if hist:
            df = pd.DataFrame(hist).groupby("date").mean(numeric_only=True).reset_index().sort_values("date")
            return df
    return None

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
    vals = [
        compute_iaqi(pm25, bp_pm25),
        compute_iaqi(pm10, bp_pm10),
        compute_iaqi(o3, bp_o3),
        compute_iaqi(no2, bp_no2),
        compute_iaqi(so2, bp_so2),
        compute_iaqi(co, bp_co)
    ]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None

def classification_aqi(model_clf, pm25, pm10, o3, no2, so2, co):
    features = np.array([[pm25, pm10, o3, no2, so2, co]])
    idx = int(model_clf.predict(features)[0])
    labels = ['Baik','Sedang','Tidak Sehat untuk Kelompok Sensitif','Tidak Sehat','Sangat Tidak Sehat','Berbahaya']
    return labels[idx]

def forecast_pm25_from_data(model_lstm, scaler_X, scaler_y, pm25_series, num_lags=4, rolling_means=(3,4)):
    if len(pm25_series) < num_lags:
        pm25_series = list(pm25_series) + [pm25_series[-1]] * (num_lags - len(pm25_series))
    lags = list(pm25_series[-num_lags:][::-1])
    preds = []
    for _ in range(4):
        rolling_features = [np.mean(lags[:n]) for n in rolling_means]
        features = np.array([[*lags, *rolling_features, datetime.utcnow().weekday(), 1 if datetime.utcnow().weekday() >=5 else 0]])
        scaled = scaler_X.transform(features)
        pred_scaled = model_lstm.predict(scaled.reshape(1, features.shape[1], 1), verbose=0)
        val = float(scaler_y.inverse_transform(pred_scaled)[0][0])
        val = float(np.clip(val, 0, 500))
        preds.append(val)
        lags = [val] + lags[:-1]
    return preds

def play_audio(path):
    if os.path.exists(path):
        data = open(path, "rb").read()
        b64 = base64.b64encode(data).decode()
        html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        st.components.v1.html(html, height=50)

# -------------------------
# SESSION STATE init
# -------------------------
if "pollutants" not in st.session_state:
    st.session_state.pollutants = None
if "loc_info" not in st.session_state:
    st.session_state.loc_info = None
if "df_hist" not in st.session_state:
    st.session_state.df_hist = None
if "preds" not in st.session_state:
    st.session_state.preds = None

# -------------------------
# Load models
# -------------------------
with st.spinner("Memuat model... (1/1)"):
    model_clf, model_lstm, scaler_X, scaler_y = load_models()

if model_clf is None:
    st.error("Model tidak ditemukan atau gagal dimuat. Pastikan folder 'models/' berisi model & scaler.")
    st.stop()

# -------------------------
# Sidebar (info & options)
# -------------------------
with st.sidebar:
    st.header("Pengaturan")
    st.write("API & Model")
    st.text_input("OpenWeather API Key", value=OPENWEATHER_API_KEY, key="owm_key_input")
    st.markdown("---")
    st.write("Aksi cepat")
    if st.button("Reset Data"):
        st.session_state.pollutants = None
        st.session_state.loc_info = None
        st.session_state.df_hist = None
        st.session_state.preds = None
        st.experimental_rerun()

# -------------------------
# Main UI — Header
# -------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<div class='card'><div class='big'>Sistem Pemantauan Kualitas Udara</div><div class='muted'>Input Manual atau Nama Kota (API) • Prediksi & Peringatan</div></div>", unsafe_allow_html=True)
with col2:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3f/Font_Awesome_5_solid_wind.svg", width=64)

# -------------------------
# Input Tabs (Manual / Kota)
# -------------------------
tab_manual, tab_api = st.tabs(["Manual", "Nama Kota (API)"])

with tab_manual:
    st.subheader("Input Manual")
    with st.form("form_manual"):
        c1, c2 = st.columns(2)
        with c1:
            pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, value=10.0, step=0.1)
            o3 = st.number_input("O₃ (µg/m³)", min_value=0.0, value=10.0, step=0.1)
            so2 = st.number_input("SO₂ (µg/m³)", min_value=0.0, value=3.0, step=0.1)
        with c2:
            pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, value=20.0, step=0.1)
            no2 = st.number_input("NO₂ (µg/m³)", min_value=0.0, value=5.0, step=0.1)
            co = st.number_input("CO (µg/m³)", min_value=0.0, value=0.5, step=0.1)
        submitted = st.form_submit_button("Simpan Input Manual")
        if submitted:
            st.session_state.pollutants = {"pm25":pm25,"pm10":pm10,"o3":o3,"no2":no2,"so2":so2,"co":co}
            st.session_state.loc_info = {"method":"manual","label":"Manual Input"}
            st.success("Data manual tersimpan. Klik 'Prediksi' untuk menjalankan model.")

with tab_api:
    st.subheader("Ambil Data Berdasarkan Nama Kota")
    city = st.text_input("Nama Kota (contoh: Jakarta)")
    if st.button("Fetch Data"):
        if not city or city.strip()=="":
            st.error("Masukkan nama kota terlebih dahulu.")
        else:
            lat, lon, label = get_coords(city)
            if lat is None:
                st.error("Gagal menemukan koordinat kota. Periksa nama kota.")
            else:
                try:
                    p = fetch_owm_realtime(lat, lon)
                    st.session_state.pollutants = p
                    st.session_state.loc_info = {"method":"api","label":label,"lat":lat,"lon":lon}
                    # try fetch historis (4 hari)
                    dfh = fetch_owm_history(lat, lon, days_back=4)
                    st.session_state.df_hist = dfh
                    st.success(f"Data berhasil diambil untuk: {label}")
                except Exception as e:
                    st.error("Gagal fetch data dari API: " + str(e))

# -------------------------
# Summary Card
# -------------------------
st.markdown("<div class='card'><div style='display:flex;justify-content:space-between;align-items:center'><div><div class='muted'>Lokasi</div><div class='big'>"
            + (st.session_state.loc_info["label"] if st.session_state.loc_info else "Belum dipilih") +
            "</div></div><div><div class='muted'>Waktu (UTC)</div><div class='small'>" + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + "</div></div></div></div>", unsafe_allow_html=True)

# -------------------------
# If pollutants exist, show table
# -------------------------
if st.session_state.pollutants:
    p = st.session_state.pollutants
    st.write("### Nilai Polutan Saat Ini (µg/m³)")
    st.table(pd.DataFrame([p], index=["Konsentrasi"]).T)

# -------------------------
# Predict button / logic
# -------------------------
if st.button("🔎 Prediksi"):
    if not st.session_state.pollutants:
        st.error("Tidak ada data polutan — isi manual atau ambil data berdasarkan nama kota.")
    else:
        p = st.session_state.pollutants
        kategori = classification_aqi(model_clf, p["pm25"], p["pm10"], p["o3"], p["no2"], p["so2"], p["co"])
        iaqi_max = compute_max_aqi(p["pm25"], p["pm10"], p["o3"], p["no2"], p["so2"], p["co"])
        # show results in card columns
        c1, c2 = st.columns([1,2])
        with c1:
            # AQI gauge (approx using iaqi_max 0-500)
            value = float(iaqi_max) if iaqi_max is not None else 0.0
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                number={'suffix':' IAQI'},
                gauge={
                    'axis': {'range':[0,500]},
                    'bar': {'color':'#1f77b4'},
                    'steps': [
                        {'range':[0,50],'color':'#50f0e6'},
                        {'range':[51,100],'color':'#50ccaa'},
                        {'range':[101,150],'color':'#f0e641'},
                        {'range':[151,200],'color':'#ff7b47'},
                        {'range':[201,300],'color':'#8f3f97'},
                        {'range':[301,500],'color':'#7e0023'}
                    ],
                },
                title={'text':"IAQI Maks (aprox)"}
            ))
            fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("<div class='card'><div class='muted'>Kategori</div><div class='big'>" + kategori + "</div><div class='muted'>Saran</div>", unsafe_allow_html=True)
            # suggestion text
            suggestion = "Tidak ada tindakan khusus."
            if kategori in ["Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]:
                suggestion = "Hentikan aktivitas luar, gunakan masker N95, dan segera cari tempat tertutup."
            elif kategori == "Tidak Sehat untuk Kelompok Sensitif":
                suggestion = "Orang sensitif disarankan membatasi aktivitas luar."
            elif kategori == "Sedang":
                suggestion = "Waspada; bagi orang sensitif pertimbangkan masker."
            st.markdown(f"<div style='margin-top:8px'>{suggestion}</div></div>", unsafe_allow_html=True)

        # play audio + animated banner for bad air
        if kategori in ["Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]:
            st.markdown("<div style='padding:10px;border-radius:8px;background:#fff3cd;color:#856404;margin-top:12px'>⚠️ <b>Perhatian:</b> Kualitas udara buruk — harap menggunakan masker.</div>", unsafe_allow_html=True)
            play_audio(ALERT_AUDIO_PATH)

        # show historis & LSTM prediction (if df_hist exists)
        if st.session_state.df_hist is not None:
            st.subheader("Data Historis (PM2.5 daily avg)")
            st.dataframe(st.session_state.df_hist)
            series = st.session_state.df_hist["pm2_5"].dropna().tolist()
            if len(series) > 0:
                preds = forecast_pm25_from_data(model_lstm, scaler_X, scaler_y, series)
                st.subheader("Prediksi PM2.5 — 4 Hari ke Depan (LSTM)")
                future_dates = [(datetime.utcnow().date() + timedelta(days=i+1)).isoformat() for i in range(len(preds))]
                dfp = pd.DataFrame({"date": future_dates, "pm2_5": preds})
                st.table(dfp)
                combined = pd.concat([st.session_state.df_hist.rename(columns={"date":"date","pm2_5":"pm2_5"}), dfp], ignore_index=True, sort=False)
                fig2 = px.line(combined, x="date", y="pm2_5", markers=True, title="Historis & Prediksi PM2.5")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Data historis tidak tersedia — gunakan mode 'Nama Kota (API)' lalu Fetch Data untuk melihat historis & prediksi LSTM.")

        # Download JSON
        res = {
            "location": st.session_state.loc_info["label"] if st.session_state.loc_info else "Manual",
            "timestamp_utc": datetime.utcnow().isoformat(),
            "pollutants": p,
            "iaqi_max": int(round(iaqi_max)) if iaqi_max is not None else None,
            "category": kategori
        }
        st.download_button("🔽 Unduh Hasil (JSON)", data=json.dumps(res, indent=2), file_name="aqi_result.json", mime="application/json")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("Fauzan Ramadhan - Universitas Gunadarma 2025.")
