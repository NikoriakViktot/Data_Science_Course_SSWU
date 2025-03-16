
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import math as mt
import os
os.environ['TCL_LIBRARY'] = r'C:\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Python313\tcl\tk8.6'




url = "http://127.0.0.1:8000/filter_telegrams/"

kharkiv = "34300"
dhipro = "34504"
chernigiv = "33135"
symu = "33275"
rivne = "33301"
zhytomyr = "33325"
kiev = "33345"
lviv = "33393"
ternopil = "33415"
Khmelnytskyi = "33429"
poltava = "33506"


payload = {
    "country_code": "ua",
    "station_id": poltava,
    "fields_to_return": ["temperature", "year", "month", "day", "hour"]
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}



# ---------------------- 1. Отримання даних через API -------------------------
url = "http://127.0.0.1:8000/filter_telegrams/"

# Видаляємо фільтр по годині, щоб отримати записи за всі години
payload = {
    "country_code": "ua",
    "station_id": kiev,
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
except requests.exceptions.RequestException as e:
    print("Помилка при виконанні запиту:", e)
    results = []

# ---------------------- 2. Обробка отриманих даних -------------------------
data_list = []  # список типу [(datetime, temperature), ...]
for record in results:
    data = record.get("data", {})
    year = data.get("year")
    month = data.get("month")
    day = data.get("day")
    hour = data.get("hour")
    temperature = data.get("temperature")
    # Переконуємося, що всі необхідні дані присутні
    if year and month and day and (hour is not None) and (temperature is not None):
        dt = datetime(year, month, day, hour)
        data_list.append((dt, temperature))

# Сортуємо записи за часом
data_list.sort(key=lambda x: x[0])

# Витягуємо температури та генеруємо часовий індекс (просто порядковий номер)
times = [x[0] for x in data_list]
temperatures = np.array([x[1] for x in data_list])
time_index = np.arange(len(temperatures))

# ---------------------- 3. Побудова моделі методом найменших квадратів (МНК) -------------------------
def MNK_Stat_characteristics(y_values):
    """
    Функція побудови поліноміальної аппроксимації другого степеня
    за методом найменших квадратів.
    Вхідний аргумент:
      y_values - масив виміряних значень.
    Повертає:
      Yout - апроксимовані значення (лінія тренду).
      C    - коефіцієнти аппроксимації.
    """
    n = len(y_values)
    Yin = np.zeros((n, 1))
    F = np.ones((n, 3))
    for i in range(n):
        Yin[i, 0] = float(y_values[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout, C

# Обчислюємо трендову лінію для температурних вимірювань
Y_trend, coefficients = MNK_Stat_characteristics(temperatures)

# ---------------------- 4. Обчислення статистичних характеристик залишків -------------------------
residuals = temperatures - Y_trend.flatten()

mean_res = np.median(residuals)
var_res = np.var(residuals)
std_res = mt.sqrt(var_res)

print("Медіана залишків:", mean_res)
print("Дисперсія залишків:", var_res)
print("СКВ залишків:", std_res)

# ---------------------- 5. Візуалізація результатів -------------------------
# Графік виміряних даних та тренду
plt.figure()
plt.plot(time_index, temperatures, label="Виміряні температури", marker='o', linestyle='-', markersize=3)
plt.plot(time_index, Y_trend, label="Тренд (МНК)", color='red', linewidth=2)
plt.xlabel("Індекс часу (порядковий номер)")
plt.ylabel("Температура")
plt.title("Температура та тренд (МНК)")
plt.legend()
plt.show()

# Гістограма залишків
plt.figure()
plt.hist(residuals, bins=20, facecolor="blue", alpha=0.5)
plt.xlabel("Залишки")
plt.ylabel("Частота")
plt.title("Гістограма залишків")
plt.show()
