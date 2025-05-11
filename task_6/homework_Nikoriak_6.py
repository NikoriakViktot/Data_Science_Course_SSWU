# --------------------------- Homework_6  ---------------------------------

"""
Виконав: Віктор Нікоряк
Homework_6, Рівень складності: II
Умови: OLAP і Data Mining з агроданими для DSS системи.

Виконання:
1. Об’єднання даних урожайності, вологості, посухи, SCOR, добрив та витрат.
2. Нормалізація та побудова інтегральної моделі SCOR.
3. Додавання додаткових показників Fert_SCOR та Cost_SCOR.
4. Збереження фінального датафрейму для візуалізації в BI-системах.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# Завантаження файлу


def calculate_scor_total(df, weights):
    """
    Розрахунок інтегрального показника SCOR_total на основі ваг і напрямку оптимізації
    :param df: pandas DataFrame з колонками
    :param weights: словник {'назва_показника': {'weight': float, 'maximize': bool}}
    """
    total_weight = sum(v['weight'] for v in weights.values())
    scor_components = []

    for col, params in weights.items():
        weight = params['weight'] / total_weight
        values = df[col]
        if not params['maximize']:
            values = 1 - values
        scor_components.append(weight * values)

    df['SCOR_total'] = sum(scor_components)
    return df


def plot_top5_crops(df):
    top5 = df.groupby('Crop_Eng')['SCOR_norm'].mean().sort_values(ascending=False).head(5)
    top5.plot(kind='barh', color='teal', title='Топ-5 культур за SCOR_norm')
    plt.xlabel('Середній SCOR_norm')
    plt.tight_layout()
    plt.show()

def plot_score_norm(df):
    plt.figure(figsize=(16, 8))
    sns.barplot(
        data=df,
        x='Region',
        y='SCOR_norm',
        hue='Crop_Eng',
        dodge=True,
        palette='viridis'
    )
    plt.xticks(rotation=90)
    plt.title("SCOR (нормалізований) по культурах і регіонах")
    plt.ylabel("SCOR_norm")
    plt.tight_layout()
    plt.show()

def plot_olap_cube(df):
    fig = px.scatter_3d(
        df,
        x='Crop_Eng',
        y='Region',
        z='SCOR_norm',
        color='Fert_SCOR',
        size=df['Cost_SCOR'].astype(float),
        title="OLAP 3D SCOR-Куб: Урожайність + Добрива + Витрати"
    )
    fig.update_traces(marker=dict(opacity=0.7))
    fig.show()

def plot_heatmap(df):
    pivot_table = df.pivot_table(values='SCOR_norm', index='Crop_Eng', columns='Region', aggfunc='mean')
    plt.figure(figsize=(16, 8))
    sns.heatmap(pivot_table,annot=True, cmap="viridis")
    plt.title("SCOR (нормалізований) по культурах і регіонах")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_cheapest_crops(df):
    best_crops = df.loc[df.groupby('Region')['Cost_SCOR'].idxmin()]
    plt.figure(figsize=(14, 6))
    sns.barplot(data=best_crops, x='Region', y='Cost_SCOR', hue='Crop_Eng')
    plt.title("Найдешевша культура по регіонах")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_crop_ratings(df):
    rating = df.groupby('Crop_Eng')[['SCOR_norm', 'Fert_SCOR', 'Cost_SCOR']].mean().sort_values('SCOR_norm')
    rating.plot(kind='barh', figsize=(10, 6), title="Середній рейтинг культур по всій Україні")
    plt.xlabel("Середній SCOR")
    plt.tight_layout()
    plt.show()


def plot_total_scor_ranking(df):
    total_rating = df.groupby('Crop_Eng')['SCOR_total'].mean().sort_values(ascending=False)
    total_rating.plot(kind='bar', figsize=(10, 6), color='darkgreen')
    plt.title("Середній SCOR_total по культурах (агро + добрива + ціна)")
    plt.ylabel("SCOR_total")
    plt.tight_layout()
    plt.show()

def plot_region_total_scor(df):
    region_avg = df.groupby('Region')['SCOR_total'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    region_avg.plot(kind='bar', color='dodgerblue')
    plt.title("Середній SCOR_total по регіонах України")
    plt.ylabel("Середній SCOR_total")
    plt.xlabel("Регіон")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_best_crop_per_region(df):
    best = df.loc[df.groupby("Region")["SCOR_total"].idxmax()]
    plt.figure(figsize=(14, 6))
    sns.barplot(data=best, x="Region", y="SCOR_total", hue="Crop_Eng")
    plt.title("Найефективніша культура в кожному регіоні (за SCOR_total)")
    plt.ylabel("SCOR_total")
    plt.xticks(rotation=90)
    plt.legend(title='Культура', loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()


# -------------------------------- БЛОК ВІЗУАЛІЗАЦІЇ --------------------------------
if __name__ == '__main__':
    df = pd.read_csv('data/final_agro_scor_dataset.csv')

    # ---------- Ваги і напрямок оптимізації ----------
    weights = {
        'SCOR_norm': {'weight': 0.5, 'maximize': True},
        'Fert_SCOR': {'weight': 0.3, 'maximize': False},
        'Cost_SCOR': {'weight': 0.2, 'maximize': False}
    }

    df = calculate_scor_total(df, weights)

    # ---------- Меню з циклом ----------
    while True:
        print("\nОберіть візуалізацію (0 — вихід):")
        print("1 - Топ-5 культур за SCOR_norm")
        print("2 - SCORE по культурах та регіонах")
        print("3 - OLAP 3D SCOR-Куб")
        print("4 - Теплова карта SCOR")
        print("5 - Найдешевша культура по регіонах")
        print("6 - Середній рейтинг культур")
        print("7 - Середній SCOR_total по культурах")
        print("8 - Середній SCOR_total по регіонах")
        print("9 - Найкраща культура по кожному регіону")
        print("10 - Змінити ваги для SCOR_total (інтегрального індексу)")

        try:
            choice = int(input("Ваш вибір: "))
        except ValueError:
            print("❌ Введіть число!")
            continue

        if choice == 0:
            print("✅ Вихід з програми.")
            break
        elif choice == 1:
            plot_top5_crops(df)
        elif choice == 2:
            plot_score_norm(df)
        elif choice == 3:
            plot_olap_cube(df)
        elif choice == 4:
            plot_heatmap(df)
        elif choice == 5:
            plot_cheapest_crops(df)
        elif choice == 6:
            plot_crop_ratings(df)
        elif choice == 7:
            plot_total_scor_ranking(df)
        elif choice == 8:
            plot_region_total_scor(df)
        elif choice == 9:
            plot_best_crop_per_region(df)
        elif choice == 10:
            try:
                print("\nВведіть нові ваги для обчислення SCOR_total:")
                print("Примітка: сума ваг буде автоматично нормалізована до 1.")
                print("Чим більша вага — тим важливіший критерій у фінальній оцінці.")

                w1 = float(input("Вага SCOR_norm (макс. врожайність): "))
                w2 = float(input("Вага Fert_SCOR (мін. добрива): "))
                w3 = float(input("Вага Cost_SCOR (мін. витрати): "))

                total = w1 + w2 + w3
                if total == 0:
                    print("Сума ваг не може бути нульовою.")
                    continue

                weights = {
                    'SCOR_norm': {'weight': w1, 'maximize': True},
                    'Fert_SCOR': {'weight': w2, 'maximize': False},
                    'Cost_SCOR': {'weight': w3, 'maximize': False}
                }

                # Нормалізація
                total_weight = sum(v['weight'] for v in weights.values())
                for key in weights:
                    weights[key]['weight'] /= total_weight

                df = calculate_scor_total(df, weights)

                print("✅ Нові нормалізовані ваги застосовано:")
                for key, value in weights.items():
                    print(f"   {key}: {value['weight']:.2f} ({'max' if value['maximize'] else 'min'})")

            except ValueError:
                print("❌ Введіть правильні числові значення.")

        else:
            print("❌ Невірний вибір. Введіть число від 0 до 10")


# ----------------------------- Висновок Домашнього Завдання №5 -----------------------------

'''
Підсумковий висновок і результати

Мета роботи:
Побудова інтегральної моделі агроефективності (SCOR_total) для підтримки прийняття рішень 
у сільському господарстві з урахуванням:
Урожайності (SCOR_norm),
Ефективності внесення добрив (Fert_SCOR),
Собівартості (Cost_SCOR).

Реалізовані кроки:
Об'єднано дані урожайності, вологості, посух, витрат і добрив.
Нормалізовано показники для порівняння культур.
Побудовано інтегральний показник SCOR_total з гнучкою зміною ваг та пріоритетів (максимізація/мінімізація).
Створено систему візуалізації, яка дозволяє:
Порівнювати культури по SCOR.
Виявляти найкращу культуру в кожному регіоні.
Аналізувати загальну ефективність по регіонах та культурах.
Змінювати ваги.

Основні спостереження:
Найефективніші культури (згідно SCOR_total):
Barley (Ячмінь) — найстабільніші результати по Україні.
Wheat (Пшениця) — добра врожайність при помірних витратах.
Soybean (Соя) — лідирує в багатьох регіонах при низькому внесенні добрив.

Найефективніші регіони (згідно середнього SCOR_total):

Volynska, Mykolaivska, Dnipropetrovska oblast — демонструють високу ефективність агровиробництва при оптимальному співвідношенні врожайності, витрат і добрив.

Інтерпретація інтегральної моделі:
SCOR_total = зважена оцінка врожайності, витрат і добрив, де:
Показники з вагою → визначають пріоритети системи підтримки рішень (DSS).
maximize=True означає, що чим більше — тим краще (наприклад, врожайність).
maximize=False означає, що чим менше — тим краще (наприклад, витрати).

'''

# --------------------------------------------------------------------------------------------------------

