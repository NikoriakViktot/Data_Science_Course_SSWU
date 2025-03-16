import math as mt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from datetime import datetime
import os

# Налаштування для Tkinter (якщо потрібно для графіків)
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
    Генерує вибірку з нормального розподілу розміром iter з параметрами:
      dm   - середнє,
      dsig - стандартне відхилення.
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
    Генерує nAV індексів аномальних вимірів (АВ) у межах [0, iter-1]
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



# ------------------------ МНК та СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ -------------------------


def MNK_Stat_characteristics(S):
    """
    Побудова поліноміальної аппроксимації (2-го порядку) за МНК
    і повернення згладжених даних Yout.
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
    Обчислює статистичні характеристики залишків (S - МНК-тренд):
    математичне сподівання, дисперсію та СКВ, після чого виводить їх.
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
    Звертається до локального бекенду http://127.0.0.1:8000/filter_telegrams/
    для отримання даних температури з вказаної станції.
    Якщо API недоступний або даних ще немає, спробує завантажити дані з CSV.
    Якщо CSV файл не існує, отримані через API дані зберігаються у CSV для подальшого використання.

    Повертає часовий індекс, температури та DataFrame.
    """
    if csv_file is None:
        csv_file = "default_temperature_data.csv"

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
    Файл повинен містити стовпці: 'year', 'month', 'day', 'hour', 'temperature'.
    Повертає часовий індекс, температури та DataFrame.
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
    Якщо CSV файл з даними існує та force_api == False, завантажує дані з нього.
    Інакше звертається до API через parse_telegrams_from_server і зберігає отримані дані у CSV.
    Повертає часовий індекс, температури та DataFrame.
    """
    if not force_api and os.path.exists(csv_file):
        print(f"Завантаження даних з CSV файлу: {csv_file}")
        return parse_temperature_csv(csv_file)
    else:
        print("Завантаження даних через API...")
        time_index, temperatures, df = parse_telegrams_from_server(country, station_id)
        return time_index, temperatures, df



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
        SV_AV = Model_NORM_AV(S0, SV, SAV, dm, dsig, Q_AV)
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
       - Дані температури отримані через API або завантажені з CSV, аналізуються методом МНК.
       
        Статистичні характеристики залишків на прикладі станції Київ
       - матиматичне сподівання ВВ= 4.27037370406738e-15
       - дисперсія ВВ = 24.67626876982095
       - СКВ ВВ= 4.9675213909776925
       
       Статистичні характеристики залишків на прикладі станції Харків
       - матиматичне сподівання ВВ= -5.124448444880856e-15
       - дисперсія ВВ = 29.146829383933678
       - СКВ ВВ= 5.39878036077906
       
       Статистичні характеристики залишків на прикладі станції Львів
       - матиматичне сподівання ВВ= 9.205027762100797e-15
       - дисперсія ВВ = 27.90893980415616
       - СКВ ВВ= 5.282891235313875

    3. Загальний висновок:
       Отримані результати свідчать про адекватність реалізованих математичних моделей. 
       Синтетична модель із заданими параметрами добре відтворює основні статистичні характеристики, 
       а додавання аномалій логічно збільшує розкид даних. Аналіз реальних даних підтверджує, 
       що побудована модель тренду адекватно апроксимує середню тенденцію температурного ряду, 
       хоча частина варіації залишається випадковою.

    Таким чином, розроблений скрипт успішно виконує вимоги завдання:
       - Моделювання випадкової величини та побудова квадратичного тренду.
       - Формування адитивної моделі (тренд + шум + аномалії).
       - Обчислення статистичних характеристик та побудова гістограм.
       - Аналіз реальних даних з можливістю завантаження через API або CSV.
    """
