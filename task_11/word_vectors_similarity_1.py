'''
Word-Vectors and Semantics:

Word2Vec — це широко використовуваний метод обробки природної мови (NLP),
який дозволяє представляти слова як вектори в безперервному векторному просторі.
Word2Vec – це спроба зіставити слова з високовимірними векторами,
щоб охопити семантичні зв’язки між словами, розроблена дослідниками Google.
Слова зі схожими значеннями повинні мати подібні векторні представлення відповідно до головного принципу Word2Vec.
Word2Vec використовує дві архітектури:

https://serokell.io/blog/word2vec
https://towardsdatascience.com/word-vectors-and-semantics-2863e7e55417
https://medium.com/nlplanet/text-similarity-with-the-next-generation-of-word-embeddings-in-gensim-466fdafa4423

'''

import spacy
import re
import warnings

warnings.filterwarnings(action='ignore')

# --------------------------------- Приклад №1 --------------------------------------

nlp = spacy.load("en_core_web_sm")


doc1 = nlp("I like salty fries and hamburgers.")
doc2 = nlp("Fast food tastes very good.")

# Подібність двох документів
print(doc1, "<->", doc2, doc1.similarity(doc2))
# Подібність токенів
french_fries = doc1[2:4]
burgers = doc1[5]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))


doc = nlp(u'The quick brown fox jumped over the lazy dogs.')
print(doc.vector)

# Створено об’єкт Doc із трьома маркерами:
tokens = nlp(u'like love hate')
# Перебір комбінацій маркерів:
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# --------------------------------- Приклад №2 --------------------------------------
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
    return str(modified_text)

# --------------- підключення словника - моделі -------------
nlp = spacy.load("uk_core_news_sm")
introduction_doc = text_filter_ukr("test_2.txt")

text_1_doc = nlp(introduction_doc)
text_2_doc = nlp("політика новини війна україна")

# text_1_doc = nlp(introduction_doc[2:5])
# text_2_doc = nlp(introduction_doc[2:5])

# Порівняння двох документів
print(text_1_doc.similarity(text_2_doc))

doc = nlp('Я хочу зелене яблуко.')
# doc = nlp("політика новини війна україна")
print(doc.similarity(doc[2:5]))