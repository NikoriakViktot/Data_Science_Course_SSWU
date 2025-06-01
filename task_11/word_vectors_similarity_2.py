'''

Word-Vectors and Semantics:

Word2Vec — це широко використовуваний метод обробки природної мови (NLP),
який дозволяє представляти слова як вектори в безперервному векторному просторі.
Word2Vec – це спроба зіставити слова з високовимірними векторами,
щоб охопити семантичні зв’язки між словами, розроблена дослідниками Google.
Слова зі схожими значеннями повинні мати подібні векторні представлення відповідно до головного принципу Word2Vec.
Word2Vec - по суті дворівнева згорткова штучна нейронна мережа.
Word2Vec використовує дві архітектури:
1) CBOW (Continuous Bag of Words);
2) Skip Gram.

https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
https://builtin.com/machine-learning/nlp-word2vec-python
https://radimrehurek.com/gensim/models/word2vec.html
https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial

'''

from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')

#  Вхідні дані
sample = open("alice.txt")
s = sample.read()

# Очищення від "\n"
f = s.replace("\n", " ")

data = []

# ітерація по файлу
for i in sent_tokenize(f):
    temp = []

    # токенізація по словам
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

# ----------- Створення векторної моделі CBOW тексту - косинусна подібність векторів = параметри 1 ---------
'''
CBOW (Continuous Bag of Words): модель CBOW передбачає поточне слово з контекстними словами в певному вікні. 
Вхідний рівень містить контекстні слова, а вихідний – поточне слово. 
Прихований шар містить розміри, які ми хочемо представити поточному слову, присутньому на вихідному шарі. 
'''
model1 = gensim.models.Word2Vec(data, min_count=1,
                                vector_size=100, window=5)

# Відображення результатів Аліса == Країна мрій
print("Cosine similarity between 'alice' " +
      "and 'wonderland' - CBOW : ",
      model1.wv.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +
      "and 'machines' - CBOW : ",
      model1.wv.similarity('alice', 'machines'))

# ---------  Створення векторної моделі Skip Gram тексту - косинусна подібність векторів = параметри 2 --------
'''
Skip Gram : Skip Gram передбачає навколишні контекстні слова в певному вікні за поточного слова. 
Вхідний рівень містить поточне слово, а вихідний – контекстні слова. 
Прихований шар містить кількість розмірів, у яких ми хочемо представити поточне слово, присутнє на вхідному шарі.
'''
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100,
                                window=5, sg=1)

# Відображення результатів Аліса == Країна мрій
print("Cosine similarity between 'alice' " +
      "and 'wonderland' - Skip Gram : ",
      model2.wv.similarity('alice', 'wonderland'))

print("Cosine similarity between 'alice' " +
      "and 'machines' - Skip Gram : ",
      model2.wv.similarity('alice', 'machines'))