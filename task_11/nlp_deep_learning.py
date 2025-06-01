'''
Аналіз тональності текстових повідомлень методами глибинного навчання:
навчання на відгуках про фільми та передбачення тональності відгуку

https://www.geeksforgeeks.org/rnn-for-text-classifications-in-nlp/
https://www.tensorflow.org/text/tutorials/text_classification_rnn
https://www.kaggle.com/code/amvillalobos/rnn-text-classification

'''


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


# ----------------------- Етап_1 дані для навчання -----------------------

# Отримайте набір даних  imdb із  tensorflow
'''
Набір даних оглядів фільмів IMDB — це набір даних для бінарної класифікації настроїв,
що містить 25 000 дуже полярних оглядів фільмів для навчання та 25 000 для тестування.
'''
dataset = tfds.load('imdb_reviews', as_supervised=True)



# -------- Етап_2 Розділення даних на test та train datasets --------------

# Розділення даних на test та train datasets
train_dataset, test_dataset = dataset['train'], dataset['test']

# Розділення даних на партії по 32 batch_size
batch_size = 32
train_dataset = train_dataset.shuffle(10000)
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
example, label = next(iter(train_dataset))

# Відображення [0] запису навчальної пари
print('Text:\n', example.numpy()[0])
print('\nLabel: ', label.numpy()[0])



# ------ Етап_3 Використання шару TextVectorization для нормалізації -----------

encoder = tf.keras.layers.TextVectorization(max_tokens=10000)
encoder.adapt(train_dataset.map(lambda text, _: text))

# Вилучення словника з шару TextVectorization
vocabulary = np.array(encoder.get_vocabulary())

# Кодування тестового прикладу та його зворотне декодування
original_text = example.numpy()[0]
encoded_text = encoder(original_text).numpy()
decoded_text = ' '.join(vocabulary[encoded_text])

print('original: ', original_text)
print('encoded: ', encoded_text)
print('decoded: ', decoded_text)

# Виправте деякі внутрішні помилки збірки
encoder._build_shapes_dict = None



# ---------------------- Етап_4 Створення моделі -------------------------------

# Створення моделі
model = tf.keras.Sequential([
	encoder,
	tf.keras.layers.Embedding(
		len(encoder.get_vocabulary()), 64, mask_zero=True),
	tf.keras.layers.Bidirectional(
		tf.keras.layers.LSTM(64, return_sequences=True)),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(1)
])

# Ініціалізація моделі
model.summary()

# Компіляція моделі
model.compile(
	loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
	optimizer=tf.keras.optimizers.Adam(),
	metrics=['accuracy']
)



# ----- Етап_5 Навчання моделі та перевірка її на тестовому наборі -------------

history = model.fit(
	train_dataset,
	epochs=1,
	validation_data=test_dataset,
)



# ------------------- Етап_6 Оцінювання точності моделі -----------------------

# Історія навчання
history_dict = history.history

# Точність навчання
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

# Втрати
loss = history_dict['loss']
val_loss = history_dict['val_loss']

# Графіки
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Accuracy', 'Validation Accuracy'])

plt.subplot(1, 2, 2)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss', 'Validation Loss'])

plt.show()



# ------------------------- Етап_7 Прогнозування_1 ------------------------------

sample_text = (
	'''The movie by Sigma Software was so good and the animation are so dope.
	I would recommend my friends to watch it.'''
)

print('sample_text_1 =', sample_text)
predictions = model.predict(np.array([sample_text]).astype(object))
print(*predictions[0])

# Рішення про тональність відгуку
if predictions[0] > 0:
	print('The review is positive')
else:
	print('The review is negative')



# ------------------------- Етап_8 Прогнозування_2 ------------------------------

import re
import spacy
from deep_translator import GoogleTranslator

# ------- фільтрація, нормалізація та частотний аналіз ------
def text_filter_ukr(filename):
    # завантаження тексту із файла
    with open(filename, encoding="utf-8") as file:
        text = file.read()
    # фільтрація від шуму - зайвих символів
    text = text.replace("\n", " ")
    text = text.replace(",", "").replace(".", "").replace("?", "").replace("!", "").replace("-", "").replace(":", "")
    text = re.sub(r'[0-9]+', '', text)
    modified_text = text.lower()
    print('modified_text =', modified_text)
    return str(modified_text[20:90])

#------------------- переклад  ----------------------------
def translator_text(text_uk):
    text_en = GoogleTranslator(source='uk', target='en').translate(text_uk)
    print('text_uk: ', text_uk)
    print('text_en: ', text_en, '\n')
    return str(text_en)

# --------------- підключення словника - моделі -------------
nlp = spacy.load("uk_core_news_sm")

text_uk = text_filter_ukr("test_2.txt")

sample_text = translator_text(text_uk)

print('sample_text_1 =', sample_text)

predictions = model.predict(np.array([sample_text]).astype(object))
print(*predictions[0])

# Рішення про тональність відгуку
if predictions[0] > 0:
	print('The review is positive')
else:
	print('The review is negative')




