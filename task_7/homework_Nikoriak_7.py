# --------------------------- Homework_  ---------------------------------

"""
Виконав: Віктор Нікоряк
Homework_7, Рівень складності: I
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
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
    print("\n📊 Топ культур у кожному кластері за середнім SCOR_total:\n")
    grouped = df_clustered.groupby(['Cluster', 'Crop_Eng'])['SCOR_total'].mean().reset_index()
    for cl in sorted(df_clustered['Cluster'].unique()):
        top = grouped[grouped['Cluster'] == cl].sort_values('SCOR_total', ascending=False).head(top_n)
        print(f"\n🔹 Кластер {cl}")
        print(top[['Crop_Eng', 'SCOR_total']])

# === ПОВНИЙ АНАЛІЗ КЛАСТЕРІВ ===

def analyze_clusters(df):
    print("\n📊 Аналіз кластерів Region × Crop:")
    clusters = cluster_by_region_crop(df)

    # Середній SCOR_total по кластерах
    avg_scor = clusters.groupby('Cluster')['SCOR_total'].mean()
    print("\n🔹 Середній SCOR_total по кластерах:")
    print(avg_scor)

    # Топ культури
    top_crops_by_cluster(clusters)

    # Heatmap
    pivot = clusters.pivot(index='Region', columns='Crop_Eng', values='SCOR_total')
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu')
    plt.title("🌡️ Heatmap SCOR_total по регіонах і культурах")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Неефективні пари
    worst = clusters.copy()
    worst['Low_SCOR_High_Cost'] = (worst['SCOR_total'] < 0.55) & (worst['Cost_SCOR'] > 0.2)
    bad_cases = worst[worst['Low_SCOR_High_Cost']]
    print("\n❌ Неефективні культура-регіон пари (низький SCOR_total < 0.55, високі витрати > 0.2):")
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
            print("❌ Введіть число!")
            continue

        if choice == 0:
            print("✅ Завершення")
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
            print("❌ Невірний вибір!")

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

# ----------------------------- Висновок Домашнього Завдання №6 -----------------------------

'''
Підсумковий висновок і результати

Мета роботи:

'''
# --------------------------------------------------------------------------------------------------------
