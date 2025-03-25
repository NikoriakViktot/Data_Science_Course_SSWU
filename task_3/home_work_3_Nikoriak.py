# --------------------------- Homework_3  ------------------------------------

'''

Виконав: Нікоряк Віктор
Homework_3,
І рівень складності:
Умови
Реалізувати рекурентне згладжування за alfa-beta фільтром для модельних вхідних
даних Дз_1 з аномаліями Дз_2, або для реальних даних. Рішення про зміст етапів підготовки
вхідних даних, розрахунок та візуалізацію показників якості процесу згладжування
прийняти самостійно.
Провести аналіз отриманих результатів та верифікацію розробленого скрипта.

ІІ рівень складності.
Реалізувати рекурентне згладжування за alfa-beta-gamma фільтром для модельних
вхідних даних Дз_1 з аномаліями Дз_2, або для реальних даних. Рішення про зміст етапів
підготовки вхідних даних, розрахунок та візуалізацію показників якості процесу
згладжування прийняти самостійно.
Провести аналіз отриманих результатів та верифікацію розробленого скрипта.

Виконання:
1. Обрати рівень складності, відкинути зайве, додати необхідне у прикладі;
2. Написати власний скрипт.

Package                      Version
---------------------------- -----------

pip                          23.1
numpy                        1.23.5
pandas                       1.5.3
xlrd                         2.0.1
matplotlib                   3.6.2

'''




import math as mt
import numpy as np
import matplotlib.pyplot as plt

import sys
import pandas as pd

import statsmodels.api as sm

from meteo_parser.telegram_filter import TelegramDataLoader


# ------------------------ ФУНКЦІЇ ДЛЯ СИНТЕТИЧНОГО МОДЕЛЮ ------------------------

def Model(n):
    """
    Генерує ідеальний квадратичний тренд.
    S0[i] = 0.0000005 * i^2.
    """
    S0 = np.zeros(n)
    for i in range(n):
        S0[i] = 0.0000005 * i * i
    return S0


def randoNORM(dm, dsig, iter):
    """
    Генерує вибірку нормальних (гаусових) шумів.
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
    Генерує індекси для аномальних вимірів із рівномірного розподілу.
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
    print('----- РІВНОМІРНИЙ розподіл індексів АВ -----')
    print(f'мат. сподівання= {mS}, дисперсія= {dS}, СКВ= {scvS}')
    print('-------------------------------------------')
    plt.hist(S, bins=20, facecolor="blue", alpha=0.5)
    plt.title("Рівномірний розподіл індексів АВ")
    plt.show()
    return SAV


def Model_NORM(SN, S0N, n):
    """
    Формує модель вимірювання: ідеальний тренд + нормальний шум.
    """
    SV = np.zeros(n)
    for i in range(n):
        SV[i] = S0N[i] + SN[i]
    return SV


def Model_NORM_AV(S0, SV, nAV, Q_AV, dsig, SAV):
    """
    Додає аномальні виміри до вибірки SV.
    Для індексів аномалій використовує попередньо згенерований SAV.
    """
    SV_AV = SV.copy()
    # Аномальні виміри з підвищеним шумом (Q_AV-подвоєння СКВ)
    SSAV = np.random.normal(0, Q_AV * dsig, nAV)
    for i in range(nAV):
        k = int(SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]
    return SV_AV


# --------------------- ФУНКЦІЇ ВИЗУАЛІЗАЦІЇ ТА ОЦІНКИ --------------------------

def Plot_AV(S0_L, SV_L, Text):
    """
    Побудова графіка: ідеальний тренд та виміряні дані.
    """
    plt.clf()
    plt.plot(SV_L, label="Виміряні дані")
    plt.plot(S0_L, label="Ідеальний тренд")
    plt.ylabel(Text)
    plt.legend()
    plt.show()


def Stat_characteristics_in(SL, Text):
    """
    Обчислює та виводить статистичні характеристики (медіана, дисперсія, СКВ) залишків
    між даними SL та апроксимованою лінією тренду (обчислюється МНК‑методом).
    """
    Yout = MNK_Stat_characteristics(SL)
    n = len(Yout)
    residuals = np.zeros(n)
    for i in range(n):
        residuals[i] = SL[i] - Yout[i, 0]
    mean_val = np.mean(residuals)
    variance_val = np.var(residuals)
    std_val = mt.sqrt(variance_val)
    print(f"Математичне сподівання залишків= {mean_val}")
    print(f"Дисперсія залишків = {variance_val}")
    print(f"СКВ залишків= {std_val}")
    plt.figure()
    plt.hist(residuals, bins=20, facecolor="blue", alpha=0.5)
    plt.title(f"Гістограма залишків: {Text}")
    plt.show()


def r2_score(SL, Yout, Text):
    """
    Розраховує коефіцієнт детермінації R² між вихідними даними SL та моделлю Yout.
    """
    n = len(Yout)
    numerator = 0
    denominator_1 = 0
    for i in range(n):
        numerator += (SL[i] - Yout[i, 0]) ** 2
        denominator_1 += SL[i]
    denominator_2 = 0
    for i in range(n):
        denominator_2 += (SL[i] - (denominator_1 / n)) ** 2
    R2 = 1 - (numerator / denominator_2)
    print('------------', Text, '-------------')
    print('Кількість елементів вибірки =', n)
    print('Коефіцієнт детермінації R² =', R2)
    return R2


def MNK_Stat_characteristics(S0):
    """
    Розраховує апроксимовану лінію тренду методом найменших квадратів (МНК)
    для подальшого аналізу залишків.
    """
    n = len(S0)
    Yin = np.zeros((n, 1))
    F = np.ones((n, 3))
    for i in range(n):
        Yin[i, 0] = float(S0[i])
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return Yout


def Sliding_Window_AV_Detect_sliding_wind(S0, n_Wind):
    """
    Виявлення та заміна аномальних вимірів за алгоритмом sliding window.
    """
    iter_total = len(S0)
    j_Wind = mt.ceil(iter_total - n_Wind) + 1
    S0_Wind = np.zeros(n_Wind)
    Midi = np.zeros(iter_total)
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = j + i
            S0_Wind[i] = S0[l]
        # Беремо медіану за вікном
        Midi[l] = np.median(S0_Wind)
    S0_Midi = np.zeros(iter_total)
    for j in range(iter_total):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi


# ------------------- РЕАЛІЗАЦІЯ α‑β ФІЛЬТРУ (KALMAN‑ПОДІБНО) -------------------

def ABF(S0):
    """
    Рекурентне згладжування α‑β фільтром.
    Вхідний сигнал S0 (масив вимірів) переводиться у форму (n,1).
    """
    N = len(S0)
    Yin = np.zeros((N, 1))
    YoutAB = np.zeros((N, 1))
    dt = 1  # крок за замовчуванням (можна адаптувати)

    # Заповнюємо вхідні дані
    for i in range(N):
        Yin[i, 0] = float(S0[i])

    # Початкові умови
    # Початкова швидкість (градієнт) оцінюється за першими двома вимірами
    speed_prev = (Yin[1, 0] - Yin[0, 0]) / dt
    x_pred = Yin[0, 0] + speed_prev * dt
    # Початкові коефіцієнти α та β (можна адаптувати або підбирати)
    alfa = 2 * (2 * 1 - 1) / (1 * (1 + 1))
    beta = 6 / (1 * (1 + 1))
    # Початкова оцінка
    YoutAB[0, 0] = Yin[0, 0] + alfa * (Yin[0, 0])

    # Рекурентний алгоритм
    for i in range(1, N):
        # Оновлення оцінки
        YoutAB[i, 0] = x_pred + alfa * (Yin[i, 0] - x_pred)
        # Оновлення швидкості (градієнту)
        speed = speed_prev + (beta / dt) * (Yin[i, 0] - x_pred)
        # Збереження швидкості для наступного кроку
        speed_prev = speed
        # Прогноз наступного виміру
        x_pred = YoutAB[i, 0] + speed_prev * dt
        # Адаптивне коригування коефіцієнтів (опційно, за i-м кроком)
        alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))
    return YoutAB

def ABF_constant(data, alpha=0.3, beta=0.1, dt=1.0):
    """
    α‑β фільтр з фіксованими (константними) alpha, beta.
    data: 1D-масив із вимірюваннями.
    alpha, beta: константи в діапазоні (0, 1).
    dt: крок часу.
    Повертає згладжений 1D-масив.
    """
    n = len(data)
    data = np.asarray(data, dtype=float)
    # Масив виходу
    Yout = np.zeros(n)

    # Початковий стан
    x_est = data[0]                     # початкова оцінка
    speed = (data[1] - data[0]) / dt    # початкова швидкість
    Yout[0] = x_est

    for i in range(1, n):
        # 1) Predict (прогноз)
        x_pred = x_est + speed * dt
        # 2) Update (оновлення)
        e = data[i] - x_pred     # похибка
        x_est = x_pred + alpha * e
        speed = speed + beta * e / dt
        Yout[i] = x_est

    return Yout
# ------------------------ БІБЛІОТЕЧНІ РЕАЛІЗАЦІЇ KALMAN ФІЛЬТРУ ------------------------

def filterpy_kalman_filter(data):
    """
    Застосування Kalman‑фільтра з бібліотеки FilterPy.
    """
    from filterpy.kalman import KalmanFilter

    dt = 1.
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.F = np.array([[1, dt],
                     [0, 1]])
    kf.H = np.array([[1, 0]])
    kf.x = np.array([[data[0]],
                     [0.]])
    kf.P = np.eye(2) * 500
    kf.Q = np.array([[1, 0],
                     [0, 1]])
    kf.R = np.array([[5]])
    filtered = []
    for z in data:
        kf.predict()
        kf.update(np.array([[z]]))
        filtered.append(kf.x[0, 0])
    return np.array(filtered)


def filterpy_kalman_forecast_6(data, forecast_horizon=6):
    """
    Прогнозуємо 6 точок (годин) уперед за допомогою FilterPy.
    1) 'Навчаємось' на всіх даних (data).
    2) Потім 6 разів викликаємо predict(), щоб отримати майбутні значення.
    Повертає (filtered, forecasts), де:
      - filtered: згладжений ряд для вхідних даних
      - forecasts: масив із 6 прогнозованих значень
    """
    from filterpy.kalman import KalmanFilter
    dt = 1.0
    kf = KalmanFilter(dim_x=2, dim_z=1)
    # Матриця переходу стану (температура + швидкість)
    kf.F = np.array([[1, dt],
                     [0, 1]])
    # Матриця спостережень
    kf.H = np.array([[1, 0]])
    # Початковий стан [температура, швидкість]
    kf.x = np.array([[data[0]],
                     [0.]])
    # Початкова коваріація
    kf.P = np.eye(2) * 500
    # Коваріації шумів
    kf.Q = np.array([[1, 0],
                     [0, 1]])
    kf.R = np.array([[5]])

    filtered = []
    # 1) "Проганяємо" Kalman-фільтр по всьому ряду
    for z in data:
        kf.predict()
        kf.update([z])  # зчитуємо вимірювання
        filtered.append(kf.x[0, 0])

    # 2) Робимо прогноз на 6 кроків (годин)
    forecasts = []
    for i in range(forecast_horizon):
        kf.predict()          # Лише predict, без update()
        forecasts.append(kf.x[0, 0])  # Беремо оцінку температури (перший елемент стану)

    return np.array(filtered), np.array(forecasts)


def pykalman_filter(data):
    from pykalman import KalmanFilter
    data = np.asarray(data, dtype=float)
    # Перетворюємо data у 2D масив (n, 1)
    observations = data.reshape(-1, 1)
    dt = 1.0

    transition_matrices = np.array([[1, dt],
                                    [0, 1]])
    observation_matrices = np.array([[1, 0]])
    initial_state_mean = [observations[0, 0], 0]

    # Передаємо коваріації як матриці, що відповідають розмірам
    observation_covariance = np.array([[5]])      # 1x1 матриця для 1D вимірювання
    transition_covariance = np.array([[1, 0],       # 2x2 матриця для 2D стану
                                       [0, 1]])

    kf = KalmanFilter(
        transition_matrices=transition_matrices,
        observation_matrices=observation_matrices,
        initial_state_mean=initial_state_mean,
        n_dim_state=2,
        n_dim_obs=1,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance
    )

    state_means, state_covs = kf.smooth(observations)
    return state_means[:, 0]
def pykalman_forecast_6(data, forecast_horizon=6):
    from pykalman import KalmanFilter
    # Робимо data як 1D numpy
    data = np.asarray(data, dtype=float)
    # Налаштовуємо KalmanFilter
    kf = KalmanFilter(
        transition_matrices=[[1, 1],
                             [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[data[0], 0],
    )
    kf = kf.em(data, n_iter=5)  # EM-алгоритм на всьому ряду

    # Застосовуємо filter() до всього ряду, щоб отримати останній стан
    filtered_state_means, filtered_state_covs = kf.filter(data)

    # Беремо останній стан
    last_state = filtered_state_means[-1]
    last_cov = filtered_state_covs[-1]

    # Покрокове "predict" на 6 кроків
    F = np.array([[1, 1],
                  [0, 1]])
    # (Для спрощення не оновлюємо P, але в реальності треба додавати Q)
    x = last_state.copy()
    forecasts = []
    for i in range(forecast_horizon):
        # predict
        x = F @ x
        forecasts.append(x[0])
    return np.array(forecasts)




# ------------------------ ФУНКЦІЯ ЗАВАНТАЖЕННЯ ДАНИХ ДЛЯ РЕАЛЬНИХ СТАНЦІЙ ------------------------

def load_station_data(stations):
    """
    Виводить список станцій, дозволяє обрати станцію та завантажити дані.
    Повертає: station_name, station_id, time_series, data_series, n.
    """
    print("Оберіть номер станції:")
    for num, (city, code) in stations.items():
        print(f"{num} - {city} (код {code})")
    station_choice = int(input("Введіть номер: "))
    if station_choice not in stations:
        print("Невірний вибір станції!")
        sys.exit(1)
    station_name, station_id = stations[station_choice]
    print(f"Обрано станцію: {station_name} (код {station_id})")
    loader_data = TelegramDataLoader(
        api_url="http://127.0.0.1:8000/filter_telegrams/",
        country_code="ua",
        station_id=station_id,
        fields_to_return=["year", "month", "day", "hour", "temperature"],
        aggregate_field="temperature"
    )
    df = loader_data.get_raw_data()
    df['dt'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df = df.sort_values('dt').reset_index(drop=True)
    df['time_idx'] = range(len(df))
    time_series = df['time_idx'].values
    data_series = df['temperature'].values
    if data_series is not None:
        loader_data.plot_data(time_series.reshape(-1, 1), data_series, title=f"Температура: {station_name}")
    n = len(data_series)
    return station_name, station_id, time_series, data_series, n

# ------------------------ ГОЛОВНИЙ БЛОК ------------------------

if __name__ == '__main__':

    print('Оберіть джерело вхідних даних та подальші дії:')
    print('1 - Модель (синтетичні дані)')
    print('2 - Реальні дані')
    Data_mode = int(input('mode: '))

    if Data_mode == 1:
        # ------------------------------ СИНТЕТИЧНА МОДЕЛЬ ------------------------------
        n = 10000                   # кількість вимірювань
        iter_val = n                # кількість реалізацій
        Q_AV = 3                    # коефіцієнт для аномалій
        nAVv = 10                   # відсоток аномальних вимірів
        nAV = int((iter_val * nAVv) / 100)  # абсолютна кількість аномалій
        dm = 0                      # математичне сподівання шуму
        dsig = 5                    # стандартне відхилення нормального шуму

        # Генерація модельних даних
        S0 = Model(n)
        SAV = randomAM(n, iter_val, nAV)
        S_noise = randoNORM(dm, dsig, iter_val)
        SV = Model_NORM(S_noise, S0, n)
        Plot_AV(S0, SV, 'Квадратична модель + нормальний шум')
        Stat_characteristics_in(SV, 'Вибірка + нормальний шум')
        SV_AV = Model_NORM_AV(S0, SV, nAV, Q_AV, dsig, SAV)
        Plot_AV(S0, SV_AV, 'Модель + нормальний шум + аномалії')
        Stat_characteristics_in(SV_AV, 'Вибірка з аномаліями')

        print('ABF: Згладжена вибірка після очищення аномалій (метод sliding window)')
        n_Wind = 5
        SV_clean = Sliding_Window_AV_Detect_sliding_wind(SV_AV, n_Wind)
        Stat_characteristics_in(SV_clean, 'Очищена вибірка (sliding window)')
        SV_filtered = ABF(SV_clean)
        r2_score(SV_clean, SV_filtered, 'α‑β фільтр (синтетичні дані)')
        plt.figure()
        plt.plot(SV_clean, label="Очищена вибірка")
        plt.plot(SV_filtered, '--', label="Згладжена (α‑β фільтр)")
        plt.xlabel("Індекс часу")
        plt.ylabel("Значення")
        plt.title("Згладжування синтетичних даних")
        plt.legend()
        plt.show()

    elif Data_mode == 2:
        # ------------------------------ РЕАЛЬНІ ДАНІ ------------------------------
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

        print("Оберіть режим фільтрації для реальних даних:")
        print("1 - Власна реалізація (α‑β фільтр)")
        print("2 - Бібліотечні реалізації Kalman‑фільтра")
        filt_mode = int(input("Введіть 1 або 2: "))

        if filt_mode == 1:
            # Власна реалізація
            station_name, station_id, time_series, data_series, n = load_station_data(stations)
            data_clean = Sliding_Window_AV_Detect_sliding_wind(data_series.copy(), n_Wind=5)
            Stat_characteristics_in(data_clean, "Очищені дані (real)")
            data_filtered = ABF(data_clean)
            r2_score(data_clean, data_filtered, "α‑β фільтр (реальні дані)")
            data_filtered_con = ABF_constant(data_clean, alpha=0.4, beta=0.1, dt=1.0)
            r2_score(data_clean, data_filtered_con.reshape(-1, 1), "α‑β фільтр (constant)")
            plt.figure()
            plt.plot(data_clean, label="Очищені дані")
            plt.plot(data_filtered, '--', label="Згладжені (α‑β фільтр)")
            plt.xlabel("Індекс часу")
            plt.ylabel("Температура")
            plt.title(f"Згладжування даних: {station_name}")
            plt.legend()
            plt.show()
            plt.figure()
            plt.plot(data_clean, label="Очищені дані")
            plt.plot(data_filtered_con, '--', label="α‑β фільтр (constant)")
            plt.xlabel("Індекс часу")
            plt.ylabel("Температура")
            plt.title(f"Згладжування даних: {station_name}")
            plt.legend()
            plt.show()
        elif filt_mode == 2:
            # Бібліотечні реалізації
            print("Оберіть бібліотечну реалізацію Kalman‑фільтра:")
            print("1 - FilterPy Kalman Filter")
            print("2 - PyKalman Kalman Filter")
            lib_choice = int(input("Введіть номер: "))
            station_name, station_id, time_series, data_series, n = load_station_data(stations)
            data_clean = Sliding_Window_AV_Detect_sliding_wind(data_series.copy(), n_Wind=5)
            Stat_characteristics_in(data_clean, "Очищені дані (real)")

            if lib_choice == 1:
                filtered_lib = filterpy_kalman_filter(data_clean)
                method_name = "FilterPy Kalman Filter"
                filtered, forecast_6 = filterpy_kalman_forecast_6(data_clean, forecast_horizon=3)
                print("Прогноз FilterPy Kalman Filter на 6 годин уперед:", forecast_6)
            elif lib_choice == 2:
                filtered_lib = pykalman_filter(data_clean)
                method_name = "PyKalman Kalman Filter"
                forecast_6 = pykalman_forecast_6(data_clean, forecast_horizon=3)
                print("Прогноз PyKalman Kalman Filter на 6 годин уперед:", forecast_6)

            else:
                print("Невірний вибір!")
                sys.exit(1)
            r2_score(data_clean, filtered_lib.reshape(-1, 1), f"{method_name} (реальні дані)")
            plt.figure()
            plt.plot(data_clean, label="Очищені дані")
            plt.plot(filtered_lib, '--', label=f"Згладжені ({method_name})")
            plt.xlabel("Індекс часу")
            plt.ylabel("Температура")
            plt.title(f"Згладжування даних: {station_name}")
            plt.legend()
            plt.show()

    else:
        print("Невірний вибір джерела!")
        sys.exit(1)


'''
Аналіз отриманих результатів - верифікація математичних моделей та результатів розрахунків.
----- РІВНОМІРНИЙ розподіл індексів АВ -----
мат. сподівання= 5049.5, дисперсія= 8310377.732345241, СКВ= 2882.772577284799
-------------------------------------------
------- НОРМАЛЬНИЙ розподіл -----
мат. сподівання= -0.04342790313554429, дисперсія= 25.148149655944888, СКВ= 5.014793082066785
---------------------------------
Математичне сподівання залишків= -9.882228368951474e-15
Дисперсія залишків = 25.137998515580342
СКВ залишків= 5.013780860346844
Математичне сподівання залишків= -1.0558665053395089e-14
Дисперсія залишків = 43.477002408943974
СКВ залишків= 6.593709305765911
ABF: Згладжена вибірка після очищення аномалій (метод sliding window)
Математичне сподівання залишків= -6.292566467891448e-15
Дисперсія залишків = 8.83320249038153
СКВ залишків= 2.9720704046811424
------------ α‑β фільтр (синтетичні дані) -------------
Кількість елементів вибірки = 10000
Коефіцієнт детермінації R² = 0.8978954241953693

Обрано станцію: Харків (код 34300)
Математичне сподівання залишків= 5.1759861076508115e-15
Дисперсія залишків = 26.902606643414362
СКВ залишків= 5.186772276032018
------------ α‑β фільтр (реальні дані) -------------
Кількість елементів вибірки = 615
Коефіцієнт детермінації R² = 0.02335970858582892
------------ α‑β фільтр (constant) -------------
Кількість елементів вибірки = 615
Коефіцієнт детермінації R² = 0.95226479652174

Обрано станцію: Харків (код 34300)

Прогноз FilterPy Kalman Filter на 6 годин уперед: [-2.31977511 -3.17514409 -4.03051307]
------------ FilterPy Kalman Filter (реальні дані) -------------
Кількість елементів вибірки = 615
Коефіцієнт детермінації R² = 0.9860238527827785

Обрано станцію: Харків (код 34300)

Прогноз PyKalman Kalman Filter на 6 годин уперед: [-1.80084837 -2.3787388  -2.95662923]
------------ PyKalman Kalman Filter (реальні дані) -------------
Кількість елементів вибірки = 615
Коефіцієнт детермінації R² = 0.9772270573587536


------------------------------------------------------------------------------------------

Висновки
------------------------------------------------------------------------------------------


1. Вибір моделі згладжування має критичне значення.  
   - Адаптивна схема для обчислення коефіцієнтів α та β (де вони зменшуються з кожним кроком) для реальних даних 
      дала дуже низький коефіцієнт детермінації (R2 = 0.023). 
      Це свідчить про те, що дана схема не відображає динаміку температурного ряду.
   - Фіксовані коефіцієнти (alpha=0.4, beta=0.1) дали значно кращий результат з (R2 = 0.95),
     що вказує на те, що для даної вибірки більш адекватно працює модель із константними параметрами.

2. Бібліотечні реалізації Kalman‑фільтра (FilterPy та PyKalman) показали ще вищі значення R2 (близько 0.98) 
   та дають дуже високий рівень згладження для реальних даних. 
   Це свідчить про те, що більш гнучкі та стабільні реалізації з оптимальним налаштуванням параметрів
   краще відтворюють динаміку процесу.

3. Синтетичні дані.  
   Для модельних даних, де використовувалася квадратична модель тренду із додаванням нормального шуму та аномалій, 
   застосування α‑β фільтру (після очищення аномалій методом sliding window) призвело до (R2 = 0.90).
   Це показує, що обрана методика згладжування досить ефективна для синтетичних даних, проте результати дещо нижчі, 
   ніж для реальних даних із застосуванням бібліотечних реалізацій.

4. Прогнозування.  
   Прогнозовані значення (на 6 годин уперед) від різних реалізацій відрізняються між собою,
   що свідчить про чутливість методів до специфікацій моделі та параметрів. 
   Наприклад, у FilterPy прогноз давав трохи нижчі значення порівняно з PyKalman, 
   що може бути зумовлено різними підходами до обчислення коваріацій шуму та початкового стану.

'''


