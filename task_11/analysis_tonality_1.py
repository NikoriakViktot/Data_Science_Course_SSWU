'''
Аналіз тональності відгуків клієнтів

pip install textblob
textblob.download_corpora

https://pythonology.eu/text-analysis-in-python-spacy-and-textblob/
https://medium.com/analytics-vidhya/how-to-translate-text-with-python-9d203139dcf5
https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair
'''


import spacy
import pandas as pd
from textblob import TextBlob

#  підключення словника - моделі
nlp = spacy.load("en_core_web_sm")

# Define a function to preprocess text using Spacy
def preprocess_text(text):
    # Токенізація тексту
    doc = nlp(text)
    # Відчищення від стопслів
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and token.text.isalpha()]
    # Перетворення токенів до рядка
    preprocessed_text = " ".join(filtered_tokens)
    return preprocessed_text

# Функція для класифікації настрою за допомогою TextBlob
def classify_sentiment(text):
    # Create a TextBlob object
    blob = TextBlob(text)
    # Get the polarity score
    polarity = blob.sentiment.polarity
    # Classify the sentiment label based on the polarity score
    if polarity > 0.3:
        sentiment_label = "positive"
    elif polarity < -0.3:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    return sentiment_label

if __name__ == '__main__':

    # Завантаження CSV файла до DataFrame
    df = pd.read_csv("reviews.csv")

    # Застосування попередньої обробки тексту та аналіз тональності до стовпця  DataFrame
    df['preprocessed_text'] = df['review'].apply(preprocess_text)
    df['sentiment_label'] = df['preprocessed_text'].apply(classify_sentiment)

    # Запис результата -  DataFrame до CSV файлу
    df.to_csv("my_processed_csv_file.csv", index=False)

    print('Результати див. в my_processed_csv_file.csv')



