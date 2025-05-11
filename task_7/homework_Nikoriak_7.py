# --------------------------- Homework_7 ---------------------------------
"""
Виконав: Віктор Нікоряк
Homework_7, Рівень складності: I
Умови: Реалізація методів кластеризаціїи.

Мета:
Розробити систему кластерного аналізу ефективності агровиробництва на основі нормалізованих показників врожайності,
витрат та застосування добрив. Створити візуалізації, heatmap та визначити неефективні культура-регіон поєднання.

Основні етапи:
1. Інтеграція агроданих (урожайність, вологість, посуха, добрива, витрати)
2. Нормалізація та побудова SCOR, Fert_SCOR, Cost_SCOR
3. Розрахунок інтегрального показника SCOR_total з вагами
4. Кластеризація: регіональна, культурна, пара Region×Crop
5. Візуалізація результатів (scatterplot, heatmap, дендрограма)
6. Побудова OLAP-запитів: топ культури, середній SCOR, неефективні поєднання
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# === КЛАСТЕРИЗАЦІЯ ===

def cluster_by_region(df, n_clusters=4):
    features = ['SCOR_norm', 'Fert_SCOR', 'Cost_SCOR']
    region_data = df.groupby("Region")[features].mean()
    scaled = StandardScaler().fit_transform(region_data)
    region_data['Cluster'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(scaled)
    return region_data.reset_index()

def cluster_by_crop(df, n_clusters=4):
    features = ['SCOR_norm', 'Fert_SCOR', 'Cost_SCOR']
    crop_data = df.groupby("Crop_Eng")[features].mean()
    scaled = StandardScaler().fit_transform(crop_data)
    crop_data['Cluster'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(scaled)
    return crop_data.reset_index()

def cluster_by_region_crop(df, n_clusters=4):
    features = ['SCOR_norm', 'Fert_SCOR', 'Cost_SCOR', 'SCOR_total']
    pair_data = df.groupby(['Region', 'Crop_Eng'])[features].mean().reset_index()
    scaled = StandardScaler().fit_transform(pair_data[features[:-1]])  # не включай SCOR_total
    pair_data['Cluster'] = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(scaled)
    return pair_data

# === ВІЗУАЛІЗАЦІЯ ===

def plot_clusters_scatter(df, x, y, group_col='Cluster', hue_col=None, title=''):
    plt.figure(figsize=(12, 6))
    if hue_col and hue_col in df.columns:
        sns.scatterplot(data=df, x=x, y=y, hue=group_col, style=hue_col, s=100)
    else:
        sns.scatterplot(data=df, x=x, y=y, hue=group_col, s=100)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plot_hierarchical_dendrogram(df, features):
    scaled = StandardScaler().fit_transform(df[features])
    linked = linkage(scaled, method='ward')
    plt.figure(figsize=(14, 7))
    dendrogram(linked, labels=df.index.tolist(), leaf_rotation=90)
    plt.title("Ієрархічна кластеризація")
    plt.tight_layout()
    plt.show()

# === ТОП КУЛЬТУРИ ПО КЛАСТЕРАХ ===

def top_crops_by_cluster(df_clustered, top_n=3):
    print("\n Топ культур у кожному кластері за середнім SCOR_total:\n")
    grouped = df_clustered.groupby(['Cluster', 'Crop_Eng'])['SCOR_total'].mean().reset_index()
    for cl in sorted(df_clustered['Cluster'].unique()):
        top = grouped[grouped['Cluster'] == cl].sort_values('SCOR_total', ascending=False).head(top_n)
        print(f"\n Кластер {cl}")
        print(top[['Crop_Eng', 'SCOR_total']])

# === ПОВНИЙ АНАЛІЗ КЛАСТЕРІВ ===

def analyze_clusters(df):
    print("\n Аналіз кластерів Region × Crop:")
    clusters = cluster_by_region_crop(df)

    # Середній SCOR_total по кластерах
    avg_scor = clusters.groupby('Cluster')['SCOR_total'].mean()
    print("\n Середній SCOR_total по кластерах:")
    print(avg_scor)

    # Топ культури
    top_crops_by_cluster(clusters)

    # Heatmap
    pivot = clusters.pivot(index='Region', columns='Crop_Eng', values='SCOR_total')
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu')
    plt.title("🌡 Heatmap SCOR_total по регіонах і культурах")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Неефективні пари
    worst = clusters.copy()
    worst['Low_SCOR_High_Cost'] = (worst['SCOR_total'] < 0.55) & (worst['Cost_SCOR'] > 0.2)
    bad_cases = worst[worst['Low_SCOR_High_Cost']]
    print("\n Неефективні культура-регіон пари (низький SCOR_total < 0.55, високі витрати > 0.2):")
    print(bad_cases[['Region', 'Crop_Eng', 'SCOR_total', 'Cost_SCOR']])

# === SCOR_total ===

def calculate_scor_total(df, weights):
    total_weight = sum(v['weight'] for v in weights.values())
    parts = []
    for col, params in weights.items():
        w = params['weight'] / total_weight
        v = df[col]
        if not params['maximize']:
            v = 1 - v
        parts.append(w * v)
    df['SCOR_total'] = sum(parts)
    return df

# === ІНТЕРАКТИВНЕ МЕНЮ ===

def interactive_menu(df):
    while True:
        print("\n Оберіть режим кластерного аналізу:")
        print("1 - Кластери по регіонах")
        print("2 - Кластери по культурах")
        print("3 - Кластери по парах (Region × Crop)")
        print("4 - Ієрархічна кластеризація по SCOR")
        print("5 - Повний аналіз кластерів (топ культури, heatmap, неефективні)")
        print("6 - Тільки топ культури по кластеру")
        print("0 - Вийти")

        try:
            choice = int(input("Ваш вибір: "))
        except ValueError:
            print("Введіть число!")
            continue

        if choice == 0:
            print("Завершення")
            break

        elif choice == 1:
            clusters = cluster_by_region(df)
            plot_clusters_scatter(clusters, 'SCOR_norm', 'Cost_SCOR', title='Кластери регіонів за ефективністю')

        elif choice == 2:
            clusters = cluster_by_crop(df)
            plot_clusters_scatter(clusters, 'SCOR_norm', 'Cost_SCOR', title='Кластери культур за ефективністю')

        elif choice == 3:
            clusters = cluster_by_region_crop(df)
            plot_clusters_scatter(clusters, 'Region', 'SCOR_total', title='Кластери Region × Crop за SCOR_total')

        elif choice == 4:
            pivot = df.groupby('Region')[['SCOR_norm', 'Fert_SCOR', 'Cost_SCOR']].mean()
            plot_hierarchical_dendrogram(pivot, ['SCOR_norm', 'Fert_SCOR', 'Cost_SCOR'])

        elif choice == 5:
            analyze_clusters(df)

        elif choice == 6:
            clusters = cluster_by_region_crop(df)
            top_crops_by_cluster(clusters)

        else:
            print("Невірний вибір!")

# === ЗАПУСК ===

if __name__ == '__main__':
    df = pd.read_csv('data/final_agro_scor_dataset.csv')

    weights = {
        'SCOR_norm': {'weight': 0.5, 'maximize': True},
        'Fert_SCOR': {'weight': 0.3, 'maximize': False},
        'Cost_SCOR': {'weight': 0.2, 'maximize': False}
    }

    df = calculate_scor_total(df, weights)
    interactive_menu(df)

# ----------------------------- Висновок Домашнього Завдання №7 -----------------------------

'''

Підсумковий висновок і результати
1. Результати кластеризації регіонів:
Кластер 0: регіони з помірною ефективністю, потребують подальшого моніторингу.
Кластери 1 і 2: регіони з найвищими показниками SCOR, що свідчить про оптимальні витрати й високу врожайність.
Кластер 3: найменш ефективні регіони, з високими витратами та низькими результатами.

2. Середній SCOR_total по кластерах (Region × Crop):

| Кластер | Середній SCOR\_total |
| ------- | -------------------- |
| 1       | ≈ 0.72               |
| 2       | ≈ 0.71               |
| 0       | ≈ 0.61               |
| 3       | ≈ 0.50               |

3. Найефективніші культури у кожному кластері:

Кластер 1: Barley, Potato, Maize* — найкраща ефективність.
Кластер 2: Wheat, Barley — стабільно високі результати.
Кластер 0: Wheat, Sunflower — середній рівень.
Кластер 3: Potato, Soybean — слабка ефективність.

4. Аналіз теплової карти (Heatmap):

Високі результати: Khersonska oblast – Sunflower, Volynska oblast – Barley.
Проблемні регіони: Vinnytska та Khmelnytska oblast — для всіх культур ефективність нижче середньої.

5. Неефективні пари (низький SCOR < 0.55, витрати > 0.2):
Особливо небезпечні:
Chernihivska – Maize (SCOR < 0.51, витрати = 1.0)
Poltavska, Sumska, Vinnytska — мають кілька культур з критичною неефективністю
Kharkivska – Sunflower, Wheat

'''