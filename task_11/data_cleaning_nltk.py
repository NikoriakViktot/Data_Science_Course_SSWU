'''
Бібліотека nltk не має вбудованої Української мови тож потребує доповнень нею.

Підготовка та аналіз даних в конвеєрі NLP:
Noise Filtering
Tokenizing
Filtering Stop Words
Frequency Analysis

Прикладний функціонал:
якій контент в домінанті у стрічці новин?

Словник з nlp:
https://medium.com/stinopys/%D1%81%D0%BB%D0%BE%D0%B2%D0%BD%D0%B8%D0%BA-nlp-b0fab1027551
Посібники:
https://dou.ua/lenta/articles/first-steps-in-nlp-nltk/
https://realpython.com/nltk-nlp-python/
https://dataknowsall.com/blog/textcleaning.html
https://spotintelligence.com/2023/09/18/top-20-essential-text-cleaning-techniques-practical-how-to-guide-in-python/
https://hex.tech/blog/Cleaning-text-data/

'''

import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.draw.dispersion import dispersion_plot
import matplotlib.pyplot as plt
import requests
from nltk.corpus import gutenberg
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer


#------------------- токенізація ----------------------------
def tokenizing_sent (example_string):
    sent = sent_tokenize(example_string)  # токенізація рядків
    print('Токенізація рядків:')
    print(sent, '\n')
    return sent

def tokenizing_word(example_string):
    word = word_tokenize(example_string)  # токенізація слів
    print('Токенізація слів:')
    print(word, '\n')
    return word

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
    return str(modified_text)

#--------------- фільтрація стоп-слів -------------------------
def filtering_stop_words(word, key):
    words_in_quote = word
    print('Фільтрація стоп-слів ВХІД:')
    print(words_in_quote)

    if key == 'english':
        print('Увага!: stopwords.words = english')
        # англомовний словник stopwords.words("english") - вбудований в nltk
        stop_words = set(stopwords.words("english"))

    else:
        print('Увага!: stopwords.words = ukrainian')
        # україномовний словник stopwords.words("ukrainian") - доповнено із зовні
        url = 'https://raw.githubusercontent.com/olegdubetcky/Ukrainian-Stopwords/main/ukrainian'
        r = requests.get(url)
        with open(nltk.data.path[0] + '/corpora/stopwords/ukrainian', 'wb') as f:
            f.write(r.content)
        stop_words = set(stopwords.words("ukrainian"))

    # збирання рядка без стопслів
    filtered_list = []
    for word in words_in_quote:
        if word.casefold() not in stop_words:
            filtered_list.append(word)
    print('Фільтрація стоп-слів ВИХІД:')
    print(filtered_list, '\n')
    return filtered_list

# --------------------- лематизація -----------------------
def lemmatize(tokens):
    lemma = WordNetLemmatizer()
    lemmatized_tokens = [lemma.lemmatize(token, pos = 'v') for token in tokens]
    print('Лематизація: ')
    print(lemmatized_tokens, '\n')
    return lemmatized_tokens

#---------------  дисперсійний аналіз токенів  -------------
def dispersion_plot_words(words, targets):
    dispersion_plot(words, targets)
    plt.show()
    return

#---------------  частотний аналіз токенів  -------------
def frequency_words(filtered_list, key):
    frequency_distribution = FreqDist(filtered_list)
    print(frequency_distribution)
    if key == 1:
        n = len(filtered_list)
        print('Увага!: маєте результати анадізу ', n, ' слів')
        print(n)
        print(frequency_distribution.most_common(n))
        frequency_distribution.plot(n, cumulative=True)
        plt.hist(filtered_list, bins=n, facecolor="blue", alpha=0.5)
        plt.show()
    else:
        n = 30
        print('Увага!: маєте результати анадізу ', n, ' слів')
        print(n)
        print(frequency_distribution.most_common(n))
        frequency_distribution.plot(n, cumulative=True)
        plt.hist(filtered_list, bins=30, facecolor="blue", alpha=0.5)
        plt.show()
        return

# ----------------------------------------- main ---------------------------------------

if __name__ == '__main__':

    #---------------------- Вихідні дані -----------------------------------
    f = open('test_2.txt', 'r', encoding="utf-8")
    example_string_0 = str(f.readlines())               # результати парсингу нових

    # --- фільтрація шуму, нормалізація та докладний частотний аналіз -----
    # example_string_0 = text_filter_ukr('test_2.txt')

    example_string_1 = """
    Muad'Dib learned rapidly because his first training was in how to learn.
    And the first lesson of all was the basic trust that he could learn.
    It's shocking to find how many people do not believe they can learn,
    and how many more believe learning to be difficult."""

    example_string_2 = "Sir, I protest. I am not a merry man!"

    example_string = example_string_0
    # example_string = example_string_1
    # example_string = example_string_2

    print('Вхідні дані:')
    print(example_string, '\n')

    # ------------------------- токенізація -------------------------------
    tokenizing_sent(example_string)
    words_in_quote = tokenizing_word(example_string)

    # ------------------- фільтрація стоп-слів ----------------------------
    # filtered_list = filtering_stop_words(words_in_quote, 'english')
    filtered_list = filtering_stop_words(words_in_quote, 'ukrainian')

    # ------------------------- лематизація -------------------------------
    filtered_list = lemmatize(filtered_list)

    # -----------------  дисперсійний аналіз токенів  ---------------------
    words = ['aa', 'aa', 'aa', 'bbb', 'cccc', 'aa', 'bbb', 'aa', 'aa', 'aa', 'cccc', 'cccc', 'cccc', 'cccc']
    targets = ['aa', 'bbb', 'f', 'cccc']
    dispersion_plot_words(words, targets)

    words = gutenberg.words("austen-sense.txt")
    targets = ["Elinor", "Marianne", "Edward", "Willoughby"]
    dispersion_plot_words(words, targets)

    words = filtered_list
    targets = ["суджа", "україни", "війська", "генштаб"]
    # targets = ["Суджа", "України", "Війська", "Генштаб"]
    dispersion_plot_words(words, targets)

    # чому графік частоти не співпадає з кількцсть повторень? - відповідь за посиланням:
    # https://medium.com/stinopys/%D1%81%D0%BB%D0%BE%D0%B2%D0%BD%D0%B8%D0%BA-nlp-b0fab1027551
    frequency_words(filtered_list, 1)

