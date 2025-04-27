import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('task_5/data/dataset_with_scor.csv')
# ------------------- 1. Створення таблиці з рейтингом -------------------

# Обчислюємо середній SCOR по кожній культурі
crop_scores = df.groupby('Crop_Eng')['SCOR'].mean().reset_index()
crop_scores = crop_scores.sort_values(by='SCOR')  # Чим менший SCOR, тим краще
crop_scores['Rank'] = range(1, len(crop_scores) + 1)  # Додаємо рейтинг

crop_scores.to_csv('crop_scor_ranking.csv', index=False)
print(" Таблиця з рейтингом культур збережена у 'crop_scor_ranking.csv'.")

# ------------------- 2. Побудова графіка: рейтинг культур -------------------

plt.figure(figsize=(10, 6))
sns.barplot(x='SCOR', y='Crop_Eng', data=crop_scores, palette='viridis', orient='h')
plt.title('Рейтинг культур за SCOR (чим нижче SCOR, тим краще)', fontsize=16)
plt.xlabel('Середній SCOR', fontsize=14)
plt.ylabel('Культура', fontsize=14)
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# ------------------- 3. Побудова графіка: залежність SCOR від вологості -------------------

# Вибираємо топ-3 найкращі культури
top_crops = crop_scores.head(3)['Crop_Eng'].tolist()

# Фільтруємо дані тільки для топ-3
df_top_crops = df[df['Crop_Eng'].isin(top_crops)]

# Побудова
plt.figure(figsize=(12, 6))
sns.lineplot(
    x='Average_soil_moisture',
    y='SCOR',
    hue='Crop_Eng',
    data=df_top_crops,
    dodge=False,
    marker='o'
)
plt.title('Залежність SCOR від вологості для топ-3 культур', fontsize=16)
plt.xlabel('Середня вологість ґрунту', fontsize=14)
plt.ylabel('SCOR', fontsize=14)
plt.grid(True)
plt.legend(title='Культура', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# ------------------- 4. Висновок -------------------

print("Топ-3 культури за результатами DSS моделі:")
for i, row in crop_scores.head(3).iterrows():
    print(f"{row['Rank']}. {row['Crop_Eng']} (середній SCOR: {row['SCOR']:.2f})")


# Зчитуємо безпосередньо наявний файл з найкращими культурами
best_crops_by_region = pd.read_csv('best_crops_under_drought.csv')

# Створюємо візуалізацію
plt.figure(figsize=(12, 6))
sns.barplot(
    x='SCOR',
    y='Region',
    hue='Crop_Eng',
    data=best_crops_by_region,
    dodge=False,
    palette="Set2"
)

plt.title('Найкращі культури для вирощування під час посухи по регіонах', fontsize=14)
plt.xlabel('SCOR (нижче значення краще)', fontsize=12)
plt.ylabel('Регіон', fontsize=12)
plt.legend(title='Культура', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Виведемо також текстовий аналіз
print("\nАналіз найкращих культур по регіонах:")
for _, row in best_crops_by_region.iterrows():
    print(f"{row['Region']}: {row['Crop_Eng']} (SCOR: {row['SCOR']:.2f})")
