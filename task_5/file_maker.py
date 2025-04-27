import pandas as pd
import glob
import os

# ---------------- 1. Завантаження маппінгу культур ----------------
crop_mapping = {
    'озима пшениця': 'Пшениця',
    'озима пшениця зр': 'Пшениця',
    'озима пшениця пар': 'Пшениця',
    'пшениця озима': 'Пшениця',
    'яра пшениця': 'Пшениця',
    'пшениця': 'Пшениця',
    'картопля': 'Картопля',
    'соя': 'Соя',
    'соняшник': 'Соняшник',
    'соняшник зр': 'Соняшник',
    'ярий ячмінь': 'Ячмінь',
    'ярий ячмінь зр': 'Ячмінь',
    'ячмінь': 'Ячмінь',
    'кукурудза': 'Кукурудза',
    'кукурудза на зерно': 'Кукурудза',
    'кукурудза на силос': 'Кукурудза'
}

# ---------------- 2. Завантаження врожайності ----------------
yield_files = glob.glob('./AgroStats_map-Урожайність-*.csv')
if not yield_files:
    raise FileNotFoundError(" Не знайдено файлів врожайності в task_5.")

print(f"Знайдено {len(yield_files)} файлів врожайності.")

yield_data_all = []

for file in yield_files:
    try:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        culture_info = filename.replace('.csv', '').split('-')[-1]
        parts = culture_info.split('_')
        crop = parts[0].lower().strip()
        year = int(parts[-1])

        if crop in crop_mapping:
            df['Crop'] = crop_mapping[crop]
            df['Year'] = year
            yield_data_all.append(df[['nameEng', 'value', 'Crop', 'Year']])
        else:
            print(f"Культуру '{crop}' пропущено — немає у маппінгу.")

    except Exception as e:
        print(f"Помилка в файлі {file}: {str(e)}")

# Об'єднуємо все
yield_data = pd.concat(yield_data_all, ignore_index=True)

# Очищення апострофів з nameEng
yield_data['nameEng'] = yield_data['nameEng'].str.replace("'", "", regex=False)

print("Підготовлені дані врожайності:")
print(yield_data.head())

# ---------------- 3. Завантаження посухи і вологості ----------------
drought_data = pd.read_csv('data/dataset_for_DCS.csv')

# ---------------- 4. З'єднання врожайності та посухи ----------------
combined = pd.merge(
    drought_data,
    yield_data,
    left_on=['Region', 'Year'],
    right_on=['nameEng', 'Year'],
    how='inner'
)

combined['Crop'] = combined['Crop_y']
combined = combined.drop(columns=['Crop_x', 'Crop_y', 'nameEng'])
combined = combined.rename(columns={'value': 'Yield'})

# Обробка пропусків у вологості
combined['Average_soil_moisture'] = combined['Average_soil_moisture'].fillna(0)

# Створення англійських назв культур
crop_eng_mapping = {
    'Картопля': 'Potato',
    'Кукурудза': 'Maize',
    'Пшениця': 'Wheat',
    'Соняшник': 'Sunflower',
    'Соя': 'Soybean',
    'Ячмінь': 'Barley'
}

combined['Crop_Eng'] = combined['Crop'].map(crop_eng_mapping)

# Створення фінального датасету
final_dataset = combined[['Region', 'Year', 'Crop', 'Crop_Eng', 'Drought', 'Average_soil_moisture', 'Yield']]

# Збереження
final_dataset.to_csv('prepared_dataset_for_voronin.csv', index=False)
print(" Фінальний датасет збережено у 'prepared_dataset_for_voronin.csv'.")
