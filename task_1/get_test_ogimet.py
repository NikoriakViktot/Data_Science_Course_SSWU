import requests
from datetime import datetime

url = "http://127.0.0.1:8000/filter_telegrams/"

# Видаляємо поле "hour" з запиту, щоб отримати записи за всі години
payload = {
    "country_code": "ua",
    "station_id": "34504",
    "fields_to_return": ["temperature", "year", "month", "day", "hour"]
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}

try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    results = response.json().get("results", [])

    print("Отримані записи:")
    for record in results:
        data = record.get("data", {})
        year = data.get("year")
        month = data.get("month")
        day = data.get("day")
        hour = data.get("hour")
        temperature = data.get("temperature")

        if year and month and day and (hour is not None):
            dt = datetime(year, month, day, hour)
            dt_str = dt.strftime("%Y-%m-%d %H:%M")
            print(f"Дата і час: {dt_str}, температура: {temperature}")
        else:
            print("Відсутні дані для формування дати:", data)

except requests.exceptions.RequestException as e:
    print("Помилка при виконанні запиту:", e)
