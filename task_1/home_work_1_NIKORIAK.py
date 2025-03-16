# --------------------------- Homework_1  ------------------------------------
"""
Виконав: Віктор Нікоряк
Homework_1, варіант 1, І рівень складності:
Закон зміни похибки – нормальний;
Закон зміни досліджуваного процесу – квадратичний.
Реальні дані – Дані температури повітря з метеостанцій.

Package                      Version
---------------------------- -----------
pip                          23.1
numpy                        1.23.5
pandas                       1.5.3
xlrd                         2.0.1
matplotlib                   3.6.2
requests                     2.32.3

"""

import math as mt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datetime import datetime
import os

# Налаштування для Tkinter (для графіків, якщо потрібно)
os.environ['TCL_LIBRARY'] = r'C:\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Python313\tcl\tk8.6'


# ------------------------ ФУНКЦІЇ ДЛЯ МОДЕЛІ ---------------------------------

def Model(n):
    """
    Побудова квадратичного тренду (ідеальної моделі процесу).

    Параметри:
      n (int): Кількість вимірювань (розмір вибірки).

    Повертає:
      np.ndarray: Масив розміром n, де кожне значення обчислено за формулою:
                  S0[i] = 0.0000005 * i^2.
    """
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = 0.0000005 * i * i
    return S0


def randoNORM(dm, dsig, iter):
    """
    Генерує вибірку випадкових чисел з нормального розподілу.

    Параметри:
      dm (float): Середнє значення (mean) розподілу.
      dsig (float): Стандартне відхилення розподілу.
      iter (int): Розмір вибірки (кількість реалізацій).

    Повертає:
      np.ndarray: Вибірка випадкових чисел з нормального розподілу.

    Додатково, функція виводить на екран медіану, дисперсію та СКВ (стандартне квадратичне відхилення),
    а також побудовує гістограму.
    """
    S = np.random.normal(dm, dsig, iter)
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    print('------- НОРМАЛЬНИЙ розподіл -----')
    print(f'мат. сподівання= {mS}, дисперсія= {dS}, СКВ= {scvS}')
    print('---------------------------------')
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.title("Нормальний розподіл")
    plt.show()
    return S


def randomAM(n, iter, nAV):
    """
    Генерує індекси для аномальних вимірів (АВ) із рівномірного розподілу.

    Параметри:
      n (int): Загальна кількість вимірювань (розмір вибірки).
      iter (int): Верхня межа для генерації чисел (наприклад, 10 000).
      nAV (int): Кількість аномальних вимірів, які потрібно згенерувати.

    Повертає:
      np.ndarray: Масив з nAV індексів, що представляють позиції аномальних вимірів.

    Функція також обчислює та виводить статистичні характеристики (медіану, дисперсію, СКВ)
    для рівномірного розподілу чисел, а також будує гістограму.
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
    """
    Формує адитивну модель, додаючи нормальний шум до квадратичного тренду.

    Параметри:
      SN (np.ndarray): Масив нормального шуму.
      S0N (np.ndarray): Масив ідеального тренду (отриманий з Model(n)).
      n (int): Кількість вимірювань.

    Повертає:
      np.ndarray: Адитивна модель, що являє собою суму тренду та шуму.
    """
    SV = np.zeros(n)
    for i in range(n):
        SV[i] = S0N[i] + SN[i]
    return SV


def Model_NORM_AV(S0, SV, nAV, Q_AV):
    """
    Додає аномальні виміри до вибірки SV.

    Параметри:
      S0 (np.ndarray): Ідеальний тренд (масив, отриманий з Model(n)).
      SV (np.ndarray): Вибірка, отримана як сума тренду та нормального шуму.
      nAV (int): Кількість аномальних вимірів, що додаються.
      Q_AV (float): Коефіцієнт, на який збільшується стандартне відхилення для аномалій.

    Повертає:
      np.ndarray: Вибірка з доданими аномальними вимірами.

    Примітка: Для визначення індексів аномалій використовується глобальний масив SAV,
             який має бути згенерований попередньо (за допомогою функції randomAM).
    """
    SV_AV = SV.copy()
    SSAV = np.random.normal(0, Q_AV * dsig, nAV)  # Зверніть увагу: тут використовується глобальний dsig
    for i in range(nAV):
        k = int(SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]
    return SV_AV


def Plot_AV(S0, SV, title_text):
    """
    Побудова графіка, що показує порівняння ідеального тренду та фактичних даних.

    Параметри:
      S0 (np.ndarray): Ідеальний тренд.
      SV (np.ndarray): Фактичні дані (вибірка).
      title_text (str): Заголовок графіка.
    """
    plt.figure()
    plt.plot(SV, label='Виміряні дані')
    plt.plot(S0, label='Тренд (ідеальна модель)')
    plt.title(title_text)
    plt.legend()
    plt.show()


# ------------------------ МНК та СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ -------------------------

def MNK_Stat_characteristics(S):
    """
    Здійснює побудову поліноміальної аппроксимації другого порядку (МНК) для вхідного масиву S.

    Параметри:
      S (np.ndarray): Вхідний масив даних (наприклад, виміряні значення).

    Повертає:
      np.ndarray: Згладжені дані (лінія тренду), отримані методом найменших квадратів.
    """
    n = len(S)
    Yin = np.zeros((n, 1))
    F = np.ones((n, 3))
    for i in range(n):
        Yin[i, 0] = float(S[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout


def Stat_characteristics(S, text=''):
    """
    Обчислює статистичні характеристики залишків між вихідними даними S та їхньою апроксимацією методом МНК.

    Параметри:
      S (np.ndarray): Вихідний масив даних.
      text (str): Текстовий опис або назва даних, що використовується для виводу.

    Повертає:
      None. Функція виводить математичне сподівання, дисперсію та СКВ залишків,
      а також відображає гістограму залишків.
    """
    Yout = MNK_Stat_characteristics(S)
    n = len(S)
    residuals = np.zeros(n)
    for i in range(n):
        residuals[i] = S[i] - Yout[i, 0]
    mean_val = np.mean(residuals)
    variance_val = np.var(residuals)
    std_val = mt.sqrt(variance_val)
    print(f"матиматичне сподівання ВВ= {mean_val}")
    print(f"дисперсія ВВ = {variance_val}")
    print(f"СКВ ВВ= {std_val}")
    print("----------------------------------")
    plt.figure()
    plt.hist(residuals, bins=20, facecolor="blue", alpha=0.5)
    plt.title(f"Гістограма залишків: {text}")
    plt.show()


# ------------------------ ФУНКЦІЇ ПАРСИНГУ ДАНИХ -----------------------------

def parse_telegrams_from_server(country, station_id, csv_file=None):
    """
    Отримує дані температури з локального бекенду через API.
    Якщо даних немає або API недоступний, функція спробує завантажити дані з CSV.
    Якщо CSV файл не існує, отримані через API дані зберігаються у CSV для подальшого використання.

    Параметри:
      country (str): Код країни (наприклад, "ua").
      station_id (str): Ідентифікатор станції.
      csv_file (str, optional): Шлях до CSV файлу для збереження/завантаження даних. За замовчуванням "default_temperature_data.csv".

    Повертає:
      tuple: (time_index, temperatures, df), де:
             - time_index (np.ndarray): Часовий індекс (послідовність чисел від 0 до кількості записів).
             - temperatures (np.ndarray): Масив значень температур.
             - df (pd.DataFrame): DataFrame з усіма отриманими даними.
    """
    if csv_file is None:
        csv_file = "default_temperature_data.csv"

    # Якщо файл існує, завантажуємо дані з CSV
    if os.path.exists(csv_file):
        print(f"Завантаження даних з CSV файлу: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print("Помилка при завантаженні CSV:", e)
            return np.array([]), np.array([]), None
        for col in ['year', 'month', 'day', 'hour']:
            df[col] = df[col].astype(int)
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
        df.sort_values(by='datetime', inplace=True)
        temperatures = df['temperature'].to_numpy()
        time_index = np.arange(len(temperatures))
        return time_index, temperatures, df

    # Якщо CSV не існує, завантажуємо дані через API
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
        return np.array([]), np.array([]), None

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
            data_list.append({
                "datetime": dt,
                "temperature": temp,
                "year": year,
                "month": month,
                "day": day,
                "hour": hour
            })
    df = pd.DataFrame(data_list)
    df.sort_values(by='datetime', inplace=True)
    temperatures = df["temperature"].to_numpy()
    time_index = np.arange(len(temperatures))
    try:
        df.to_csv(csv_file, index=False)
        print(f"Дані збережено у файл: {csv_file}")
    except Exception as e:
        print("Помилка при збереженні CSV:", e)
    return time_index, temperatures, df


def parse_temperature_csv(file_name):
    """
    Завантажує CSV файл з даними про температуру.

    Параметри:
      file_name (str): Шлях до CSV файлу. Файл повинен містити стовпці:
                       'year', 'month', 'day', 'hour', 'temperature'.

    Повертає:
      tuple: (time_index, temperatures, df), де:
             - time_index (np.ndarray): Часовий індекс.
             - temperatures (np.ndarray): Масив температур.
             - df (pd.DataFrame): DataFrame із завантаженими даними.
    """
    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        print("Помилка завантаження CSV:", e)
        return np.array([]), np.array([]), None
    required_cols = ['year', 'month', 'day', 'hour', 'temperature']
    for col in required_cols:
        if col not in df.columns:
            print(f"Стовпець {col} відсутній у файлі!")
            return np.array([]), np.array([]), None
    for col in ['year', 'month', 'day', 'hour']:
        df[col] = df[col].astype(int)
    df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.sort_values(by='datetime', inplace=True)
    temperatures = df['temperature'].to_numpy()
    time_index = np.arange(len(temperatures))
    print("Дані завантажено з файлу:", file_name)
    return time_index, temperatures, df


def get_temperature_data(country, station_id, csv_file=None, force_api=False):
    """
    Завантажує дані температури або через CSV, або через API, залежно від наявності файлу та параметра force_api.

    Параметри:
      country (str): Код країни (наприклад, "ua").
      station_id (str): Ідентифікатор станції.
      csv_file (str, optional): Шлях до CSV файлу для збереження/завантаження даних.
                                За замовчуванням: "default_temperature_data.csv".
      force_api (bool, optional): Якщо True, примусово завантажує дані через API, ігноруючи наявність CSV файлу.
                                   За замовчуванням: False.

    Повертає:
      tuple: (time_index, temperatures, df), де:
             - time_index (np.ndarray): Часовий індекс.
             - temperatures (np.ndarray): Масив температур.
             - df (pd.DataFrame): DataFrame з отриманими даними.
    """
    if not force_api and os.path.exists(csv_file):
        print(f"Завантаження даних з CSV файлу: {csv_file}")
        return parse_temperature_csv(csv_file)
    else:
        print("Завантаження даних через API...")
        time_index, temperatures, df = parse_telegrams_from_server(country, station_id)
        return time_index, temperatures, df


# ------------------------ ГОЛОВНИЙ БЛОК --------------------------------------

if __name__ == '__main__':
    # Список доступних станцій
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
    print("2 - Реальні дані (парсинг через API або CSV)")
    choice_data = int(input("Введіть 1 або 2: "))

    if choice_data == 1:
        # Параметри синтетичної моделі
        n = 10000
        dm = 0
        dsig = 5
        Q_AV = 3
        nAV_percent = 10
        nAV = int(n * nAV_percent / 100)

        S0 = Model(n)
        S = randoNORM(dm, dsig, n)
        SAV = randomAM(n, n, nAV)
        SV = Model_NORM(S, S0, n)
        Plot_AV(S0, SV, "Квадратичний тренд + нормальний шум")
        Stat_characteristics(SV, "Вибірка з нормальним шумом")
        SV_AV = Model_NORM_AV(S0, SV, nAV, Q_AV)
        Plot_AV(S0, SV_AV, "Квадратичний тренд + нормальний шум + аномалії")
        Stat_characteristics(SV_AV, "Вибірка з аномаліями")

    elif choice_data == 2:
        print("Оберіть спосіб завантаження реальних даних:")
        print("1 - Завантаження через API")
        print("2 - Завантаження з CSV файлу")
        method = int(input("Введіть 1 або 2: "))
        if method == 1:
            print("Оберіть номер станції:")
            for num, (city, code) in stations.items():
                print(f"{num} - {city} (код {code})")
            station_choice = int(input("Введіть номер: "))
            if station_choice not in stations:
                print("Невірний вибір станції!")
            else:
                station_name, station_id = stations[station_choice]
                print(f"Обрано станцію: {station_name} (код {station_id})")
                # force_api=True примусово завантажує дані через API (ігноруючи CSV)
                time_index, temperatures, df = get_temperature_data("ua", station_id, force_api=True)
                if len(temperatures) == 0:
                    print("Немає даних для цієї станції.")
                else:
                    plt.figure()
                    plt.plot(time_index, temperatures, label="Температура (виміряна)")
                    Yout = MNK_Stat_characteristics(temperatures)
                    plt.plot(time_index, Yout, 'r', label="Тренд (МНК)")
                    plt.xlabel("Часовий індекс")
                    plt.ylabel("Температура")
                    plt.title(f"Температура та тренд: {station_name}")
                    plt.legend()
                    plt.show()
                    Stat_characteristics(temperatures, f"Температура (залишки) - {station_name}")
        elif method == 2:
            file_name = input("Введіть шлях до CSV файлу (або залиште порожнім для дефолтного): ").strip()
            if file_name == "":
                file_name = "default_temperature_data.csv"
            time_index, temperatures, df = parse_temperature_csv(file_name)
            if len(temperatures) == 0:
                print("Не вдалося завантажити дані з файлу.")
            else:
                plt.figure()
                plt.plot(time_index, temperatures, label="Температура (з CSV)")
                Yout = MNK_Stat_characteristics(temperatures)
                plt.plot(time_index, Yout, 'r', label="Тренд (МНК)")
                plt.xlabel("Часовий індекс")
                plt.ylabel("Температура")
                plt.title("Температура та тренд (з CSV)")
                plt.legend()
                plt.show()
                Stat_characteristics(temperatures, "Температура (залишки) з CSV файлу")
        else:
            print("Невірний вибір способу завантаження даних!")
    else:
        print("Невірний вибір джерела даних!")

    # --------------------- ВИСНОВКИ ----------------------
    """
    ВИСНОВКИ:

    1. Синтетична модель:
       - Було згенеровано 10 000 випадкових значень за нормальним розподілом 
         (середнє ≈0, СКВ ≈5).
       - Побудовано квадратичний тренд, до якого додано нормальний шум.
       - Статистичні характеристики залишків (розраховані методом МНК) показали, що 
         математичне сподівання залишків практично дорівнює нулю, дисперсія становить 
         приблизно 24.90, а СКВ – близько 4.99.
       - Додавання аномалій суттєво збільшує розкид залишків (дисперсія ≈45.30, СКВ ≈6.73),
         що свідчить про вплив викидів на загальний розкид даних.

    2. Реальні дані:
       - Дані температури, отримані через API або завантажені з CSV, аналізуються методом МНК.

       Статистичні характеристики залишків (метод МНК) на прикладі:
       - Станції Київ:
         • матиматичне сподівання ВВ= 4.27×10⁻¹⁵,
         • дисперсія ВВ = 24.68,
         • СКВ ВВ= 4.97.
       - Станції Харків:
         • матиматичне сподівання ВВ= -5.12×10⁻¹⁵,
         • дисперсія ВВ = 29.15,
         • СКВ ВВ= 5.40.
       - Станції Львів:
         • матиматичне сподівання ВВ= 9.21×10⁻¹⁵,
         • дисперсія ВВ = 27.91,
         • СКВ ВВ= 5.28.

    3. Загальний висновок:
       Отримані результати демонструють, що реалізовані моделі є адекватними:
       - Синтетична модель правильно відтворює задані параметри (нормальний шум і квадратичний тренд),
         а додавання аномалій логічно збільшує розкид даних.
       - Аналіз реальних даних показує, що побудована модель тренду методом МНК
         добре апроксимує середню тенденцію температурного ряду, хоча певна частина варіації залишається випадковою.

    Таким чином, скрипт успішно задовольняє вимоги завдання:
       - Моделювання випадкової величини та побудова квадратичного тренду.
       - Формування адитивної моделі (тренд + шум + аномалії).
       - Обчислення статистичних характеристик і побудова гістограм.
       - Аналіз реальних даних із можливістю завантаження через API або CSV.
    """
