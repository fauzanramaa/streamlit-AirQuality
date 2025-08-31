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