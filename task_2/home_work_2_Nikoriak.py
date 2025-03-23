# --------------------------- Homework_2  ------------------------------------
"""
Виконав: Віктор Нікоряк
Homework_2, варіант 1, І рівень складності:
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
import sys

from meteo_parser.telegram_filter import TelegramDataLoader


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
    SSAV = np.random.normal(0, Q_AV * dsig, nAV)
    for i in range(nAV):
        k = int(SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]
    return SV_AV

# ----- Коефіцієнт детермінації - оцінювання якості моделі --------
def r2_score(SL, Yout, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    iter = len(Yout)
    numerator = 0
    denominator_1 = 0
    for i in range(iter):
        numerator = numerator + (SL[i] - Yout[i, 0]) ** 2
        denominator_1 = denominator_1 + SL[i]
    denominator_2 = 0
    for i in range(iter):
        denominator_2 = denominator_2 + (SL[i] - (denominator_1 / iter)) ** 2
    R2_score_our = 1 - (numerator / denominator_2)
    print('------------', Text, '-------------')
    print('кількість елементів вбірки=', iter)
    print('Коефіцієнт детермінації (ймовірність апроксимації)=', R2_score_our)

    return R2_score_our



# ------------------------ МНК та СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ -------------------------

def MNK(SO):
    """
    Побудова регресійної моделі (поліноміальна апроксимація другого порядку)
    для вхідного масиву S за методом найменших квадратів (МНК).

    Повертає:
      np.ndarray: Згладжені дані (лінію тренду), отриману за допомогою МНК.
    """
    n = len(SO)
    Yin = np.zeros((n, 1))
    F = np.ones((n, 3))
    for i in range(n):
        Yin[i, 0] = float(SO[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    print("Регресійна модель:")
    print(f"y(t) = {C[0, 0]} + {C[1, 0]} * t + {C[2, 0]} * t^2")
    return Yout


# ------------------------ МНК детекція та очищення АВ ------------------------------
def MNK_AV_Detect(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    return C[1, 0]


# ---------------------------  МНК ПРОГНОЗУВАННЯ -------------------------------
def MNK_Extrapol(S0, koef):
    iter = len(S0)
    Yout_Extrapol = np.zeros((iter + koef, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    print('Регресійна модель:')
    print('y(t) = ', C[0, 0], ' + ', C[1, 0], ' * t', ' + ', C[2, 0], ' * t^2')
    for i in range(iter + koef):
        Yout_Extrapol[i, 0] = C[0, 0] + C[1, 0] * i + (C[2, 0] * i * i)
    return Yout_Extrapol

# ------------------------------ Виявлення АВ за алгоритмом medium -------------------------------------
def Sliding_Window_AV_Detect_medium(S0, n_Wind, Q):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    # -------- еталон  ---------
    j = 0
    for i in range(n_Wind):
        l = (j + i)
        S0_Wind[i] = S0[l]
        dS_standart = np.var(S0_Wind)
        scvS_standart = mt.sqrt(dS_standart)
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        mS = np.median(S0_Wind)
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
        # --- детекція та заміна АВ --
        if scvS > (Q * scvS_standart):
            # детектор виявлення АВ
            S0[l] = mS
    return S0

# ------------------------------ Виявлення АВ за МНК -------------------------------------
def Sliding_Window_AV_Detect_MNK(S0, Q, n_Wind):
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros(n_Wind)
    # Використовуємо S0, що передається як параметр:
    Speed_standart = MNK_AV_Detect(S0)
    Yout_S0 = MNK(S0)
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = j + i
            S0_Wind[i] = S0[l]
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
        Speed_standart_1 = abs(Speed_standart * mt.sqrt(iter))
        Speed_1 = abs(Q * Speed_standart * mt.sqrt(n_Wind) * scvS)
        if Speed_1 > Speed_standart_1:
            S0[l] = Yout_S0[l, 0]
    return S0


# ------------------------------ Виявлення АВ за алгоритмом sliding window -------------------------------------
def Sliding_Window_AV_Detect_sliding_wind(S0, n_Wind):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    Midi = np.zeros((iter))
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        Midi[l] = np.median(S0_Wind)
    # ---- очищена вибірка  -----
    S0_Midi = np.zeros((iter))
    for j in range(iter):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi


# ----- статистичні характеристики лінії тренда  --------

def MNK_Stat_characteristics(S0):
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
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


# ----- статистичні характеристики екстраполяції  --------
def Stat_characteristics_extrapol(koef, SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    #  довірчий інтервал прогнозованих значень за СКВ
    scvS_extrapol = scvS * koef
    print('------------', Text, '-------------')
    print('кількість елементів ивбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('Довірчий інтервал прогнозованих значень за СКВ=', scvS_extrapol)
    print('-----------------------------------------------------')
    return




# ----- статистичні характеристики вхідної вибірки  --------
def Stat_characteristics_in(SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))
    for i in range(iter):
        SL0[i] = SL[i] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print('------------', Text, '-------------')
    print('кількість елементів вбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('-----------------------------------------------------')
    return


# ----- статистичні характеристики лінії тренда  --------
def Stat_characteristics_out(SL_in, SL, Text):
    # статистичні характеристики вибірки з урахуванням тренду
    Yout = MNK_Stat_characteristics(SL)
    iter = len(Yout)
    SL0 = np.zeros((iter))
    for i in range(iter):
        SL0[i] = SL[i, 0] - Yout[i, 0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    # глобальне лінійне відхилення оцінки - динамічна похибка моделі
    Delta = 0
    for i in range(iter):
        Delta = Delta + abs(SL_in[i] - Yout[i, 0])
    Delta_average_Out = Delta / (iter + 1)
    print('------------', Text, '-------------')
    print('кількість елементів ивбірки=', iter)
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    print('Динамічна похибка моделі=', Delta_average_Out)
    print('-----------------------------------------------------')
    return



# --------------- графіки тренда, вимірів з нормальним шумом  ---------------------------

def Plot_AV(S0_L, SV_L, Text):
    """
    Побудова графіка, що показує порівняння ідеального тренду та фактичних даних.

    Параметри:
      S0_L (np.ndarray): Ідеальний тренд.
      SV_L (np.ndarray): Фактичні дані (вибірка).
      title_text (str): Заголовок графіка.
    """
    plt.clf()
    plt.plot(SV_L)
    plt.plot(S0_L)
    plt.ylabel(Text)
    plt.show()
    return



# ------------------------ ГОЛОВНИЙ БЛОК --------------------------------------
# Глобальні змінні для синтетичної моделі
dsig = 5   # стандартне відхилення нормального шуму (синтетична модель)
if __name__ == '__main__':
    # Список доступних станцій для реальних даних (API)
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

    print('Оберіть джерело вхідних даних та подальші дії:')
    print('1 - Модель (синтетичні дані)')
    print('2 - Реальні дані (парсинг через API або CSV)')
    print('3 - Бібліотеки для статистичного навчання -->>> STOP')
    Data_mode = int(input('mode: '))

    if Data_mode == 1:
        # Синтетична модель
        n = 10000
        iter_val = n      # кількість реалізацій
        Q_AV = 3          # коефіцієнт для аномалій
        nAVv = 10         # відсоток аномальних вимірів
        nAV = int((iter_val * nAVv) / 100)
        dm = 0

        # Генеруємо синтетичні дані
        S0 = Model(n)
        SAV = randomAM(n, iter_val, nAV)
        S = randoNORM(dm, dsig, n)
        SV = Model_NORM(S, S0, n)
        Plot_AV(S0, SV, "Квадратичний тренд + нормальний шум")
        Stat_characteristics(SV, "Вибірка з нормальним шумом")
        SV_AV = Model_NORM_AV(S0, SV, nAV, Q_AV)
        Plot_AV(S0, SV_AV, "Квадратичний тренд + нормальний шум + аномалії")
        Stat_characteristics(SV_AV, "Вибірка з аномаліями")
        # Для синтетичної моделі використовуємо дані SV_AV
        data_series = SV_AV
        time_series = np.arange(len(data_series))
    elif Data_mode == 2:
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
                sys.exit(1)
            else:
                station_name, station_id = stations[station_choice]
                print(f"Обрано станцію: {station_name} (код {station_id})")
                loader_data = TelegramDataLoader(
                    api_url="http://127.0.0.1:8000/filter_telegrams/",
                    country_code="ua",
                    station_id=station_id,
                    fields_to_return=["year", "month", "day", "hour", "temperature"],
                    aggregate_field="temperature")
                time_idx, values, df = loader_data.get_daily_data(force_api=True)
                print(df)

                if values is not None:
                    loader_data.plot_data(time_idx, values, title=f"Середньодобові дані")

                plt.figure()
                plt.plot(time_idx, values, label="Температура (середньодобова)")
                Yout = MNK_Stat_characteristics(values)
                plt.plot(time_idx, Yout, 'r', label="Тренд (МНК)")
                plt.xlabel("Часовий індекс (дні)")
                plt.ylabel("Температура")
                plt.title(f"Температура та тренд: {station_name}")
                plt.legend()
                plt.show()
                Stat_characteristics(values, f"Температура (залишки) - {station_name}")
                data_series = values
                time_series = time_idx
        elif method == 2:
            file_name = input("Введіть шлях до CSV файлу (або залиште порожнім для дефолтного): ").strip()
            if file_name == "":
                file_name = "default_data.csv"
            csv_loader = TelegramDataLoader(csv_file=file_name)
            time_series, temperatures, df = csv_loader.get_daily_data()
            print("Кількість температурних значень:", len(temperatures))
            plt.figure()
            plt.plot(time_series, temperatures, label="Температура (з CSV)")
            Yout = MNK_Stat_characteristics(temperatures)
            plt.plot(time_series, Yout, 'r', label="Тренд (МНК)")
            plt.xlabel("Часовий індекс (дні)")
            plt.ylabel("Температура")
            plt.title("Температура та тренд (з CSV)")
            plt.legend()
            plt.show()
            Stat_characteristics(temperatures, "Температура (залишки) з CSV файлу")
            data_series = temperatures
            time_series = time_series
        else:
            print("Невірний вибір способу завантаження даних!")
            sys.exit(1)
    elif Data_mode == 3:
        print('Бібліотеки Python для реалізації методів статистичного навчання:')
        print('https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html')
        print('https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html')
        print('https://scikit-learn.org/stable/modules/sgd.html#regression')
        sys.exit(0)
    else:
        print("Невірний вибір джерела даних!")
        sys.exit(1)

    # ------------------------- Функціонал очищення та навчання -------------------------
    print('Оберіть функціонал процесів навчання:')
    print('1 - Детекція та очищення від АВ: метод medium')
    print('2 - Детекція та очищення від АВ: метод MNK')
    print('3 - Детекція та очищення від АВ: метод sliding window')
    print('4 - MNK згладжування')
    print('5 - MNK прогнозування')
    mode = int(input('mode: '))

    if mode == 1:
        N_Wind_Av = 5   # розмір ковзного вікна
        Q = 1.6         # коефіцієнт виявлення
        S_detect = Sliding_Window_AV_Detect_medium(data_series.copy(), N_Wind_Av, Q)
        Stat_characteristics_in(S_detect, 'Вибірка очищена (medium)')
        Yout_clean = MNK(S_detect)
        Stat_characteristics(data_series - Yout_clean.flatten(), 'Залишки (medium)')
        Plot_AV(Model(len(data_series)), S_detect, 'Очищена вибірка (medium)')
    elif mode == 2:
        N_Wind = 5
        Q_MNK = 7
        S_detect = Sliding_Window_AV_Detect_MNK(data_series.copy(), Q_MNK, N_Wind)
        Stat_characteristics_in(S_detect, 'Вибірка очищена (MNK)')
        Yout_clean = MNK(S_detect)
        Stat_characteristics(data_series - Yout_clean.flatten(), 'Залишки (MNK)')
        Plot_AV(Model(len(data_series)), S_detect, 'Очищена вибірка (MNK)')
    elif mode == 3:
        N_Wind = 5
        S_detect = Sliding_Window_AV_Detect_sliding_wind(data_series.copy(), N_Wind)
        Stat_characteristics_in(S_detect, 'Вибірка очищена (sliding window)')
        Yout_clean = MNK(S_detect)
        Stat_characteristics(data_series - Yout_clean.flatten(), 'Залишки (sliding window)')
        Plot_AV(Model(len(data_series)), S_detect, 'Очищена вибірка (sliding window)')
    elif mode == 4:
        N_Wind = 5
        S_detect = Sliding_Window_AV_Detect_sliding_wind(data_series.copy(), N_Wind)
        Stat_characteristics_in(S_detect, 'Вибірка очищена (sliding window)')
        Yout_smooth = MNK(S_detect)
        Stat_characteristics(data_series - Yout_smooth.flatten(), 'Залишки (згладжування)')
        r2 = r2_score(data_series, Yout_smooth, "MNK_модель_згладжування")
        print("R2 score для згладженої моделі:", r2)
        Plot_AV(Model(len(data_series)), S_detect, 'MNK згладжування')
    elif mode == 5:
        # MNK прогнозування
        N_Wind = 5
        koef_Extrapol = 0.2  # прогноз на 20% від розміру вибірки
        n = len(data_series)
        koef = int(mt.ceil(n * koef_Extrapol))  # кількість точок прогнозування
        S_detect = Sliding_Window_AV_Detect_sliding_wind(data_series.copy(), N_Wind)
        Stat_characteristics(S_detect, 'Вибірка очищена (sliding window)')
        Yout_forecast = MNK_Extrapol(S_detect, koef)
        Stat_characteristics_extrapol(koef, Yout_forecast, 'MNK прогнозування')
        plt.figure()
        # Побудова графіку: реальні дані та прогноз
        plt.plot(range(len(data_series)), data_series, label='Реальні дані')
        plt.plot(range(len(Yout_forecast)), Yout_forecast, label='Прогноз (МНК)')
        plt.xlabel("Індекс часу")
        plt.ylabel("Температура")
        plt.title("Реальні дані та прогноз (МНК)")
        plt.legend()
        plt.show()
    else:
        print("Невірний вибір функціоналу навчання!")
        sys.exit(1)


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
