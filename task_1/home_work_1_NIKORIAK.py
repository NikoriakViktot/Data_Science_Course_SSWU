import math as mt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datetime import datetime

import os
os.environ['TCL_LIBRARY'] = r'C:\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Python313\tcl\tk8.6'


# ------------------------ ФУНКЦІЇ ДЛЯ МОДЕЛІ ---------------------------------


def Model(n):
    """Квадратичний тренд (ідеальна модель процесу)."""
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = 0.0000005 * i * i
    return S0

def randoNORM(dm, dsig, iter):
    """
    Генерація випадкової величини з нормальним розподілом.
    Повертає масив розміром `iter` з параметрами:
      dm   - середнє (mean)
      dsig - стандартне відхилення
    """
    S = np.random.normal(dm, dsig, iter)
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('------- НОРМАЛЬНИЙ розподіл -----')
    print(f'мат. сподівання= {mS}, дисперсія= {dS}, СКВ= {scvS}')
    print('---------------------------------')
    # гістограма
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.title("Нормальний розподіл")
    plt.show()
    return S

def randomAM(n, iter, nAV):
    """
    Генерує `nAV` індексів аномальних вимірів (АВ) у межах від 0 до iter-1
    та виводить статистику рівномірно згенерованого масиву S.
    """
    SAV = np.zeros(nAV)
    S = np.zeros(n)
    for i in range(n):
        S[i] = np.random.randint(0, iter)
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    for i in range(nAV):
        SAV[i] = mt.ceil(np.random.randint(1, iter))

    print('номери АВ: SAV=', SAV)
    print('----- РІВНОМІРНИЙ розподіл індексів АВ -----')
    print(f'мат. сподівання= {mS}, дисперсія= {dS}, СКВ= {scvS}')
    print('-------------------------------------------')
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.title("Рівномірний розподіл індексів АВ")
    plt.show()
    return SAV

def Model_NORM(SN, S0N, n):
    """Формує суму тренду та нормальних похибок."""
    SV = np.zeros(n)
    for i in range(n):
        SV[i] = S0N[i] + SN[i]
    return SV

def Model_NORM_AV(S0, SV, SAV, dm, dsig, Q_AV):
    """
    Додає аномальні виміри в SV на індексах, заданих SAV.
    Кожен аномальний вимір = S0[k] + нормальний шум із дисперсією, збільшеною в Q_AV разів.
    """
    SV_AV = SV.copy()
    nAV = len(SAV)
    SSAV = np.random.normal(dm, Q_AV * dsig, nAV)
    for i in range(nAV):
        k = int(SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]
    return SV_AV

def Plot_AV(S0, SV, title_text):
    """Відображає два графіки: модельний тренд S0 та вибірку SV."""
    plt.figure()
    plt.plot(SV, label='Виміряні дані')
    plt.plot(S0, label='Тренд (ідеальна модель)')
    plt.title(title_text)
    plt.legend()
    plt.show()


# ------------------------ МНК та СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ ------------------


def MNK_Stat_characteristics(S):
    """
    Побудова поліноміальної аппроксимації (2-го порядку) за МНК
    і повернення вектора Yout (згладжені дані).
    """
    n = len(S)
    Yin = np.zeros((n, 1))
    F = np.ones((n, 3))
    for i in range(n):
        Yin[i, 0] = float(S[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i*i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout

def Stat_characteristics(S, text=''):
    """
    Виводить статистичні характеристики (медіана, дисперсія, СКВ) для різниці
    між S та його МНК-трендом.
    """
    Yout = MNK_Stat_characteristics(S)
    n = len(S)
    residuals = np.zeros(n)
    for i in range(n):
        residuals[i] = S[i] - Yout[i, 0]

    mS = np.median(residuals)
    dS = np.var(residuals)
    scvS = mt.sqrt(dS)
    print(f"--- {text} ---")
    print(f"Медіана залишків: {mS}")
    print(f"Дисперсія залишків: {dS}")
    print(f"СКВ залишків: {scvS}")
    print("----------------------------------")

    # побудова гістограми залишків
    plt.figure()
    plt.hist(residuals, bins=20, facecolor="blue", alpha=0.5)
    plt.title(f"Гістограма залишків: {text}")
    plt.show()

# ------------------------ ПАРСИНГ ДАНИХ З СЕРВЕРА (ТЕЛЕГРАМИ) ----------------


def parse_telegrams_from_server(country, station_id):
    """
    Звертається до локального бекенду http://127.0.0.1:8000/filter_telegrams/
    для отримання температури з вказаної станції.
    Повертає (time_index, temperatures) як масив NumPy.
    """
    url = "http://127.0.0.1:8000/filter_telegrams/"
    payload = {
        "country_code": country,
        "station_id": station_id,
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
        return np.array([]), np.array([])

    data_list = []
    for record in results:
        data = record.get("data", {})
        year = data.get("year")
        month = data.get("month")
        day = data.get("day")
        hour = data.get("hour")
        temp = data.get("temperature")
        if year and month and day and (hour is not None) and (temp is not None):
            dt = datetime(year, month, day, hour)
            data_list.append((dt, temp))

    # Сортуємо за датою
    data_list.sort(key=lambda x: x[0])
    temperatures = np.array([x[1] for x in data_list])
    time_index = np.arange(len(temperatures))
    return time_index, temperatures



if __name__ == '__main__':
    # Список доступних станцій (можна розширити)
    stations = {
        1: ("Харків", "34300"),
        2: ("Дніпро", "34504"),
        3: ("Чернігів", "33135"),
        4: ("Суми", "33275"),
        5: ("Рівне", "33301"),
        6: ("Житомир", "33325"),
        7: ("Київ", "33345"),
        8: ("Львів", "33393"),
        9: ("Тернопіль", "33415"),
        10: ("Хмельницький", "33429"),
        11: ("Полтава", "33506")
    }

    print("Оберіть джерело даних:")
    print("1 - Синтетична модель (квадратичний тренд + нормальний шум + аномалії)")
    print("2 - Реальні дані (парсинг станції через локальний бекенд)")
    choice_data = int(input("Введіть 1 або 2: "))

    if choice_data == 1:
        # ПАРАМЕТРИ МОДЕЛІ
        n = 10000
        dm = 0
        dsig = 5
        Q_AV = 3
        nAV_percent = 10
        nAV = int(n * nAV_percent / 100)

        # 1) Ідеальний тренд
        S0 = Model(n)
        # 2) Нормальний шум
        S = randoNORM(dm, dsig, n)
        # 3) Аномальні індекси
        SAV = randomAM(n, n, nAV)
        # 4) Адитивна модель (тренд + шум)
        SV = Model_NORM(S, S0, n)
        Plot_AV(S0, SV, "Квадратичний тренд + нормальний шум")
        Stat_characteristics(SV, "Вибірка з нормальним шумом")

        # 5) Додаємо аномалії
        SV_AV = Model_NORM_AV(S0, SV, SAV, dm, dsig, Q_AV)
        Plot_AV(S0, SV_AV, "Квадратичний тренд + нормальний шум + аномалії")
        Stat_characteristics(SV_AV, "Вибірка з аномаліями")

    elif choice_data == 2:
        # Вибір станції
        print("Оберіть номер станції:")
        for num, (city, code) in stations.items():
            print(f"{num} - {city} (код {code})")
        station_choice = int(input("Введіть номер: "))

        if station_choice not in stations:
            print("Невірний вибір станції!")
        else:
            station_name, station_id = stations[station_choice]
            print(f"Обрано станцію: {station_name} (код {station_id})")

            # Парсимо дані температури з локального сервера
            time_index, temperatures = parse_telegrams_from_server(country="ua", station_id=station_id)
            if len(temperatures) == 0:
                print("Немає даних для цієї станції.")
            else:
                # Будуємо тренд методом МНК
                plt.figure()
                plt.plot(time_index, temperatures, label="Температура (виміряна)")
                # МНК-тренд
                Yout = MNK_Stat_characteristics(temperatures)
                plt.plot(time_index, Yout, 'r', label="Тренд (МНК)")
                plt.title(f"Температура та тренд: {station_name}")
                plt.legend()
                plt.show()

                # Виводимо статистичні характеристики
                Stat_characteristics(temperatures, f"Температура (залишки) - {station_name}")

    else:
        print("Невірний вибір!")

    """
    Висновки

    1. Синтетична модель:
    - Було створено 10 000 випадкових значень із нормальним розподілом (середнє близько 0, стандартне відхилення близько 5). 
      Потім додано квадратичний тренд та випадкові аномалії.
    - Статистичні характеристики показали, що створений шум добре відповідає заданим параметрам.
    - Введення аномалій суттєво збільшило дисперсію, що видно на графіках у вигляді значних відхилень.
    
    2. Для реальних даних станції Харків (код 34300):
    - Побудували квадратичний тренд за реальними даними температур.
    - Аналіз залишків показав, що медіана залишків близька до нуля (≈0.34), отже, модель добре відображає середню температуру.
    - Дисперсія (≈29.15) та стандартне відхилення залишків (≈5.40) досить високі, це говорить про значні коливання температур, які модель не пояснює повністю.
    - Гістограма залишків дозволила оцінити рівномірність їх розподілу, вказуючи на випадковий характер похибок.
    
    Отже, виконано наступні завдання:
    - Створення моделі випадкових значень (нормальний розподіл).
    - Побудова квадратичної моделі (тренд).
    - Формування адитивної моделі (тренд + шум + аномалії).
    - Обчислення статистичних характеристик.
    - Аналіз реальних температурних даних та їх порівняння з модельними розрахунками.
    - Додатково реалізовано отримання реальних даних та візуалізацію результатів.


    """
