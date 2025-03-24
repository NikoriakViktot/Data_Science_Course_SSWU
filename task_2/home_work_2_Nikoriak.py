# --------------------------- Homework_2  ------------------------------------
"""
Виконав: Віктор Нікоряк
Homework_2,
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
import datetime
from datetime import date, datetime
import os
import sys
import pandas as pd

from scipy.signal import detrend
from scipy.signal import periodogram
from scipy.optimize import curve_fit


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler



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
    # print('номери АВ: SAV=', SAV)
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


def compare_library_model(data_series, time_series, degree=2, model_type="SGDRegressor"):
    """
    Функція для побудови тренду за допомогою бібліотечної моделі (sklearn) з поліноміальними ознаками
    та порівняння результату з  методом МНК.

    Параметри:
      data_series: вектор вихідних даних (наприклад, температури).
      time_series: часовий індекс або послідовність вимірювань (якщо це дати, вони будуть перетворені у числа).
      degree: степінь поліноміальної апроксимації (за замовчуванням 2).
      model_type: рядок з назвою моделі (наприклад, "LinearRegression", "Ridge", "Lasso", "ElasticNet", "SGDRegressor").

    Повертає:
      y_pred_library: прогнозовані значення, отримані бібліотечною моделлю.
      y_pred_mnk: прогнозовані значення за вашим методом МНК.

    Побудова графіка дає можливість візуально порівняти вихідні дані, бібліотечну модель та МНК.
    """
    # Перетворюємо часовий індекс у числовий формат
    if isinstance(time_series[0], (date, datetime)):
        t = np.array([x.toordinal() for x in time_series]).reshape(-1, 1)
    else:
        t = np.array(time_series).reshape(-1, 1)

    y = np.array(data_series)

    # Якщо обрана модель SGDRegressor, використовуємо конвеєр із PolynomialFeatures та StandardScaler
    if model_type == "SGDRegressor":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.linear_model import SGDRegressor

        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=True),
            StandardScaler(),
            SGDRegressor(loss="squared_error", max_iter=2000, tol=1e-4)
        )
        model.fit(t, y.ravel())
        y_pred_library = model.predict(t)
    else:
        from sklearn.preprocessing import PolynomialFeatures
        if model_type == "LinearRegression":
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_type == "Ridge":
            from sklearn.linear_model import Ridge
            model = Ridge()
        elif model_type == "Lasso":
            from sklearn.linear_model import Lasso
            model = Lasso()
        elif model_type == "ElasticNet":
            from sklearn.linear_model import ElasticNet
            model = ElasticNet()
        else:
            print("Невідома модель. Використовується LinearRegression за замовчуванням.")
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

        poly = PolynomialFeatures(degree=degree, include_bias=True)
        t_poly = poly.fit_transform(t)
        model.fit(t_poly, y)
        y_pred_library = model.predict(t_poly)

    y_pred_mnk = MNK(data_series)

    # Побудова графіку для порівняння результатів
    plt.figure(figsize=(10, 5))
    plt.plot(t, y, label="Вихідні дані", alpha=0.5)
    plt.plot(t, y_pred_library, label=f"Бібліотечна модель: {model_type}", linewidth=2)
    plt.plot(t, y_pred_mnk, label="Модель МНК (custom)", linestyle='--', linewidth=2)
    plt.xlabel("Часовий індекс")
    plt.ylabel("Температура")
    plt.title("Порівняння бібліотечної моделі та МНК (custom)")
    plt.legend()
    plt.show()

    return y_pred_library, y_pred_mnk

def MNK_Extrapol_sin_cos (S0, koef):
    iter = len(S0)
    Yout_Extrapol = np.zeros((iter + koef, 1))
    YReal = np.zeros(((iter + koef), 1))
    YMNK = np.zeros(((iter + koef), 1))
    Yout = np.zeros((iter, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 4))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    a0 = C[0, 0]
    b0 = C[1, 0] / (mt.sqrt(abs(C[2, 0] / C[0, 0])))
    w0 = mt.sqrt(abs(C[2, 0] / C[0, 0]))
    for i in range(iter):
        Yout[i, 0] = (a0 * mt.cos(w0 * i) + b0 * mt.sin(w0 * i))
    print('Ідеальна модель тренда: y(t) = ', 7.8 , ' *  cos (', 0.05, ' * t)', ' + ', 9.5, ' * sin(', w0, ' * t )')
    print('Регресійна модель_МНК: y(t) = ', C[0, 0], ' + ', C[1, 0], ' * t', ' + ', C[2, 0], ' * t^2', '+ ', C[3, 0], ' * t^3')
    print('Регресійна сінусно-косинусна модель: y(t) = ', a0, ' *  cos (', w0, ' * t)', ' + ', b0, ' * sin(', w0, ' * t )')
    for i in range(iter+koef):
        Yout_Extrapol[i, 0] = (a0 * mt.cos(w0 * i) + b0 * mt.sin(w0 * i))   # проліноміальна крива МНК - прогнозування
        YReal[i, 0] = (7.8 * mt.cos(0.05 * i) + 9.5 * mt.sin(0.05 * i))  # ідеальна крива - вхідна
        YMNK[i, 0] = C[0, 0] + C[1, 0] * i + (C[2, 0] * i * i)  # проліноміальна крива МНК
    plt.plot(Yin,  label="time series")
    plt.plot(YReal, 'r--', label="perfect trend")
    plt.plot(Yout_Extrapol, label="LSM sin-cos R&D model")
    plt.legend()
    plt.ylabel('Динаміка нелінійного процесу: екстраполяція')
    plt.savefig('MNK_Extrapol_sin_cos.png')
    plt.savefig('MNK_Extrapol_sin_cos.jpg')
    plt.show()

    return Yout_Extrapol

# ------------------------------ МНК sin_cos -------------------------------------
def MNK_sin_cos (S0):
    iter = len(S0)
    Yout = np.zeros((iter, 1))
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 4))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    a0 = C[0, 0]
    b0 = C[1, 0] / (mt.sqrt(abs(C[2, 0] / C[0, 0])))
    w0 = mt.sqrt(abs(C[2, 0] / C[0, 0]))
    for i in range(iter):
        Yout[i, 0]=(a0 * mt.cos(w0 * i) + b0 * mt.sin(w0 * i))
    print('Регресійна сінусно-косинусна модель:')
    print('y(t) = ', a0, ' *  cos (', w0, ' * t)', ' + ', b0, ' * sin(', w0, ' * t )')

    return Yout

def fit_exponential(values):
    """Оцінка параметрів експоненційної моделі y = a * exp(b*t) через scipy.curve_fit"""
    def exp_model(t, a, b): return a * np.exp(b * t)
    popt, _ = curve_fit(exp_model, np.arange(len(values)), values, p0=(values[0], 0.01))
    return popt

def forecast_exponential(values, horizon):
    """Побудова прогнозу експоненційної моделі на horizon точок"""
    a, b = fit_exponential(values)
    t = np.arange(len(values) + horizon)
    return a * np.exp(b * t)



def estimate_period(y):
    # Частота дискретизації = 1 день
    freqs, power = periodogram(y, fs=1)
    # Виключаємо нульову частоту і беремо максимум
    idx = np.argmax(power[1:]) + 1
    return 1 / freqs[idx]
def rolling_period(y, window=None):
    periods = []
    for start in range(len(y) - window):
        segment = y[start:start+window]
        periods.append(estimate_period(segment))
    return np.array(periods)
def r2_score_1d(SL, Y_pred, Text):
    """
    Тут SL i Y_pred — обидва 1D numpy arrays однакової довжини.
    """
    n = len(Y_pred)
    numerator = 0
    denominator_2 = 0
    mean_SL = np.mean(SL)
    for i in range(n):
        numerator += (SL[i] - Y_pred[i])**2
        denominator_2 += (SL[i] - mean_SL)**2
    R2_score_our = 1 - (numerator / denominator_2)
    print(f"------------ {Text} -------------")
    print(f"Кількість елементів вибірки= {n}")
    print(f"R² = {R2_score_our}")
    return R2_score_our


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
    print('2 - Реальні дані (парсинг через API)')
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
            df = loader_data.get_raw_data()
            df['dt'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            df = df.sort_values('dt').reset_index(drop=True)
            df['time_idx'] = range(len(df))
            time_series = df['time_idx'].values.reshape(-1, 1)
            data_series = df['temperature'].values
            if data_series is not None:
                loader_data.plot_data(time_series, data_series, title=f" дані")
            n = len(time_series)

        print("Оберіть спосіб моделювання  реальних даних:")
        print("1 - МНК реалізація")
        print('2 - Бібліотеки для статистичного навчання')
        print('3 - Нелінійна екстрополяція')
        print('4 - Нелінійна sin - cos')



        method = int(input("Введіть цифру: "))

        if method == 1:
                plt.figure()
                plt.plot(time_series, data_series, label="Температура (середньодобова)")
                Yout = MNK_Stat_characteristics(data_series)
                plt.plot(time_series, Yout, 'r', label="Тренд (МНК)")
                plt.xlabel("Часовий індекс (дні)")
                plt.ylabel("Температура")
                plt.title(f"Температура та тренд: {station_name}")
                plt.legend()
                plt.show()
                Stat_characteristics(data_series, f"Температура (залишки) - {station_name}")

        if method == 2:
            time_series = np.arange(len(time_series))
            # y_lib, y_mnk = compare_library_model(values, time_series, degree=2, model_type="Ridge")
            # y_lib, y_mnk = compare_library_model(values, time_series, degree=2, model_type="LinearRegression")
            y_lib, y_mnk = compare_library_model(data_series, time_series, degree=5, model_type="SGDRegressor")
            sys.exit(1)
        if method == 3:
            # Перевірка даних і чистка (sliding window)
            time_series = np.arange(len(data_series))
            n = len(data_series)
            y = data_series.copy()
            t = np.arange(len(y)).reshape(-1, 1)

            train_y = y[:500]
            train_t = t[:500]
            test_y = y[500:]
            test_t = t[500:]

            omega = 2 * np.pi / n
            y_detr = detrend(y)
            t_future = np.arange(n, n + 7).reshape(-1, 1)

            model = make_pipeline(
                FunctionTransformer(lambda X: np.column_stack([np.sin(omega * X.ravel()),
                                                               np.cos(omega * X.ravel())]), validate=False),
                PolynomialFeatures(degree=5, include_bias=True),
                StandardScaler(),
                Ridge(alpha=1.0)
            )
            model.fit(train_t, train_y)

            y_pred = model.predict(test_t)

            print("Test R²:", r2_score(test_y.reshape(-1, 1), y_pred.reshape(-1, 1), "MNK_модель_SinCos Polynomial"))
            print("Test MSE:", mean_squared_error(test_y, y_pred))
            print("Test MAE:", mean_absolute_error(test_y, y_pred))

            plt.figure()
            plt.plot(train_t, train_y, label="Train ")
            plt.plot(test_t, test_y, label="Actual days)")
            plt.plot(test_t, y_pred, '--', label="Forecast")
            plt.xlabel("Day index")
            plt.ylabel("Temperature")
            plt.title("Train vs Forecast on last 7 days")
            plt.legend()
            plt.show()

            t = np.arange(len(y)).reshape(-1, 1)
            model = make_pipeline(
                FunctionTransformer(lambda X: np.column_stack([np.sin(omega * X.ravel()),
                                                               np.cos(omega * X.ravel())]), validate=False),
                PolynomialFeatures(degree=5, include_bias=True),
                StandardScaler(),
                Ridge(alpha=1.0)
            )

            model.fit(t, y)
            y_pred = model.predict(t)
            r2 = r2_score(data_series, y_pred.reshape(-1, 1), "MNK_модель_SinCos Polynomial")
            print("R2 score для SinCos Polynomial моделі:", r2)

            # 5) Побудова графіку
            plt.figure()
            plt.plot(t, y, label="Очищені дані (detrended)")
            plt.plot(t, y_pred, label=f"SinCos Polynomial (5th)")
            plt.xlabel("Часовий індекс")
            plt.ylabel("Температура")
            plt.title("5‑й порядок Sin‑Cos поліноміальної регресії")
            plt.legend()
            plt.show()
            print("MSE:", mean_squared_error(data_series, y_pred.reshape(-1, 1)))
            print("MAE:", mean_absolute_error(data_series, y_pred.reshape(-1, 1)))

            # 3) Cross‑validation (5 folds)
            scores = cross_val_score(model, t, data_series, cv=5, scoring="r2")
            print("CV R² scores:", scores)
            print("Mean CV R²:", scores.mean())
            y_future = model.predict(t_future)


            cleaned = Sliding_Window_AV_Detect_sliding_wind(data_series.copy(), n_Wind=600)
            Stat_characteristics_in(cleaned, 'Очистка (sliding window)')

            # Експоненційна модель + прогноз
            a, b = fit_exponential(cleaned)
            print(f"Параметри експоненційної моделі: a={a:.4f}, b={b:.4f}")
            y_exp = forecast_exponential(cleaned, horizon=int(0.5 * len(cleaned)))

            # Sin‑Cos MNK тренд + прогноз
            trend_sin = MNK_sin_cos(cleaned)
            y_sin = MNK_Extrapol_sin_cos(cleaned, koef=int(0.5 * len(cleaned)))

            # Побудова графіку
            plt.figure()
            plt.plot(time_series, cleaned, label="Очищені дані")
            plt.plot(np.arange(len(y_exp)), y_exp, label="Прогноз експоненційної")
            plt.plot(np.arange(len(trend_sin)), trend_sin, '--', label="Sin‑Cos тренд")
            plt.plot(np.arange(len(y_sin)), y_sin, label="Sin‑Cos прогноз")
            plt.xlabel("Індекс часу")
            plt.ylabel("Значення")
            plt.title("Нелінійна екстраполяція")
            plt.legend()
            plt.show()
            sys.exit(0)
        if method == 4:
            n = len(data_series)
            train_size = int(0.8 * n)
            X_train = time_series[:train_size].reshape(-1, 1)
            y_train = data_series[:train_size]
            X_test = time_series[train_size:].reshape(-1, 1)
            y_test = data_series[train_size:]
            omega = 2 * np.pi / n

            model = make_pipeline(
                FunctionTransformer(lambda X: np.column_stack([np.sin(omega * X.ravel()),
                                                               np.cos(omega * X.ravel())]), validate=False),
                PolynomialFeatures(degree=5, include_bias=True),
                StandardScaler(),
                LinearRegression()
            )

            # 3) Навчаємося лише на train
            model.fit(X_train, y_train)

            # 4) Прогнозуємо на test
            y_pred_test = model.predict(X_test)

            # 5) Оцінюємо похибку (тільки на test, бо це out‑of‑sample)
            mse = mean_squared_error(y_test, y_pred_test)
            r2 = r2_score_1d(y_test, y_pred_test, "MNK_модель_SinCos Polynomial")
            print("Test MSE:", mse)
            print("Test R²:", r2)

            # 6) Візуалізація
            plt.figure(figsize=(10, 5))
            plt.plot(X_train, y_train, label="Train", color='blue')
            plt.plot(X_test, y_test, label="Actual (test)", color='orange')
            plt.plot(X_test, y_pred_test, '--', label="Forecast", color='green')
            plt.xlabel("Time index")
            plt.ylabel("Temperature")
            plt.title("Train vs Forecast (out‑of‑sample)")
            plt.legend()
            plt.show()
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
        koef_Extrapol = 0.5
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
'''Висновоки:

1. Якість моделі та роль очищення даних.  
   Очищення від аномальних вимірів (методи medium, MNK, sliding window) відчутно впливає на статистику залишків і загальний \(R^2\). 
   Вибір конкретного методу залежить від природи вхідних даних та рівня шуму: чим «брудніші» дані, тим більше користі від попередньої обробки.

2. Поліноміальні та sin‑cos моделі.  
   - Квадратичні (2-го порядку) і поліноми вищого порядку (наприклад, 5‑го) дають різні результати, 
     причому високий порядок може «перевчитися» (overfitting), якщо вибірка мала.  
   - Нелінійні sin‑cos моделі краще вловлюють сезонність, що підтверджує вищий (R2) (до 0.65–0.71),
     але тільки за умови достатнього охоплення даних і коректного визначення періоду.

3. Обмеження за обсягом даних.  
   - На невеликих вибірках (зокрема коли є лише декілька місяців спостережень) модель «підганяється» під короткострокові коливання.
     Це призводить до низького чи навіть від’ємного R2.  
   - Для довгострокового прогнозу та визначення кліматичної норми бажано мати мінімум 30 років історичних даних, 
     щоб врахувати всі сезонні та багаторічні тренди.

4. Перевірка моделей (out‑of‑sample, крос‑валідація).  
   - При розбитті на train/test і перевірці «на майбутніх точках» (які модель «не бачила») часто спостерігається падіння R2. 
     Це свідчить про необхідність більш адаптивних методів або більшої вибірки.  
   - Крос‑валідація також показує, що модельна похибка і значення R2 можуть істотно змінюватися залежно від конкретних фрагментів даних.

5. Практичний підсумок.  
   - Найкращі результати отримують тоді, коли враховано реальну сезонність (sin‑cos підхід) і модель навчається на достатньо великому масиві даних.  
   - При цьому очищення від аномалій і грамотне розбиття на train/test дозволяють оцінити справжню здатність моделі до передбачення.  
   - На маленьких часових інтервалах похибка прогнозу зростає, оскільки модель не бачить повноцінної картини сезонних і багаторічних коливань.
   '''
