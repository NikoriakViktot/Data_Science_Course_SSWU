'''
Бібліотека spacy - встановлення та завантаження
pip install spacy
Встановлення словника - корпусу Української мови / Англійської:
spacy download uk_core_news_sm
spacy download en_core_web_sm

Пісібник:
https://realpython.com/natural-language-processing-spacy-python/
https://spacy.io/usage/linguistic-features

Моделі spacy:
https://spacy.io/models

'''

import spacy
import pathlib
import re
from collections import Counter
import matplotlib.pyplot as plt
from nltk.draw.dispersion import dispersion_plot
from nltk import FreqDist

# --------------- підключення словника - моделі -------------
nlp = spacy.load("uk_core_news_sm")

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
    words = modified_text.split()
    print('modified_text =', modified_text)
    words.sort()
    words_dict = dict()
    for word in words:
        if word in words_dict:
            words_dict[word] = words_dict[word] + 1
        else:
            words_dict[word] = 1
    print("Кількість слів: %d" % len(words))
    print("Кількість унікальних слів: %d" % len(words_dict))
    print("Усі використані слова:")
    for word in words_dict:
        print(word.ljust(20), words_dict[word])
    return nlp(modified_text)


#---------------  дисперсійний аналіз токенів  -------------
def dispersion_plot_words(words, targets):
    dispersion_plot(words, targets)
    plt.show()
    return

#---------------  частотний аналіз токенів/лем  -------------
def frequency_words(token_lemma_list):
    n = len(token_lemma_list)
    n = 10
    frequency_distribution = FreqDist(token_lemma_list)
    frequency_distribution.plot(n, cumulative=True)
    plt.hist(token_lemma_list, bins=n, facecolor="blue", alpha=0.5)
    plt.show()
    return

#---------------------- Вихідні дані ---------------------------

introduction_doc = nlp("Обробка природньої мови.")
print(type(introduction_doc))                               # особливості роботи із даними - doc - документ

file_name = "test_2.txt"
introduction_doc = nlp(pathlib.Path(file_name).read_text(encoding="utf-8"))

# ------- фільтрація, нормалізація та частотний аналіз ------
introduction_doc = text_filter_ukr("test_2.txt")

#------------------- токенізація ----------------------------
token_word = [token.text for token in introduction_doc]     # особливості обробки даних
print(type(token_word))
print(token_word)

#--------------- фільтрація стоп-слів ------------------------
token_word_is_stop = [token for token in introduction_doc if not token.is_stop]
print(token_word_is_stop)

# --------------------- лематизація -----------------------
token_lemma_list = []
for token in token_word_is_stop:
    if str(token) != str(token.lemma_):
        print(f"{str(token):>20} : {str(token.lemma_)}")
        token_lemma_list.append(str(token.lemma_))
print('token_lemma_list: ', token_lemma_list)


#------------------  частотний аналіз лем  ----------------
complete_doc = nlp(str(token_lemma_list))
words = [token.text for token in complete_doc if token.is_stop != True and token.is_punct != True]
word_freq = Counter(words)
common_words = word_freq.most_common(len(word_freq))
print(common_words)

frequency_words(token_lemma_list)

words = token_lemma_list
print(common_words[0][0])
targets = [common_words[0][0], common_words[1][0], common_words[2][0], common_words[3][0]]
dispersion_plot_words(words, targets)

