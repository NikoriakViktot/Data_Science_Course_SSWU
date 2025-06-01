'''
Для аналізу тональності використано декілька бібліотек: vaderSentiment, textblob
Особливість: переклад оригіналу до англомовного сегменту текста.

pip install textblob
textblob.download_corpora

https://pythonology.eu/text-analysis-in-python-spacy-and-textblob/
https://medium.com/analytics-vidhya/how-to-translate-text-with-python-9d203139dcf5
https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair
'''

import spacy
from textblob import TextBlob
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------- підключення словника - моделі -------------
nlp = spacy.load("en_core_web_sm")

#------------------- токенізація ----------------------------
def preprocess_text(text):
    # Токенізація
    doc = nlp(text)
    # Видалення стопслів
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and token.text.isalpha()]
    # Перетворення токенів до рядка
    preprocessed_text = " ".join(filtered_tokens)
    return preprocessed_text

#------------------- переклад  ----------------------------
def translator_text(text_uk):
    text_en = GoogleTranslator(source='auto', target='en').translate(text_uk)
    print('text_uk: ', text_uk)
    print('text_en: ', text_en, '\n')
    return text_en

#---------- визначення тональності текста -----------------

def analysis_tonality_blob_1(text):
    # Очищення тексту
    preprocessed_text = preprocess_text(text)
    # Створення екземпляру / моделі TextBlob
    blob = TextBlob(preprocessed_text)
    # Отримання полярності тексту між -1 (негативний) і 1 (позитивний)
    polarity = blob.sentiment.polarity
    print('Результат blob_1: ')
    print('polarity = ', polarity)
    if polarity > 0.3:
        print("Positive", '\n')
    elif polarity < -0.3:
        print("Negative", '\n')
    else:
        print("Neutral", '\n')
    return

def analysis_tonality_blob_2(text):
    testimonial = TextBlob(text)
    print('Результат blob_2: ')
    print(testimonial.text, '\n')
    return

def analysis_tonality_vaderSentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    print('Результат vaderSentiment: ')
    print("{:-<65} {}".format(text, str(vs)), '\n')
    return

if __name__ == '__main__':

    # ---------------- Вихідні дані ------------------------
    text_uk = "В Україні триває жахлива війна, людей вбивають, скрізь трупи"

    # text_uk = "Їжа була жахлива"

    text_en = translator_text(text_uk)

    # text_en = preprocess_text(text_en)

    analysis_tonality_blob_1(text_en)

    analysis_tonality_blob_1(text_en)

    analysis_tonality_vaderSentiment(text_en)





