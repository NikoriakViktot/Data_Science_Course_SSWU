
import pandas as pd

# Завантаження даних
df = pd.read_csv('task_5/data/dataset_with_scor.csv')

# Фільтруємо тільки випадки, де була посуха
drought_data = df[df['Drought'] == 1]

# Групуємо по регіону і культурі: обчислюємо середній SCOR
region_crop_scor = drought_data.groupby(['Region', 'Crop_Eng'])['SCOR'].mean().reset_index()

# Для кожного регіону знаходимо культуру з найнижчим SCOR
best_crops_by_region = region_crop_scor.loc[region_crop_scor.groupby('Region')['SCOR'].idxmin()]

# Сортуємо за алфавітом регіони для красивого вигляду
best_crops_by_region = best_crops_by_region.sort_values('Region')

# Збережемо в таблицю
best_crops_by_region.to_csv('best_crops_under_drought.csv', index=False)

print("Таблиця з кращими культурами під час посухи створена: 'best_crops_under_drought.csv'.")

# Показати
print(best_crops_by_region)
