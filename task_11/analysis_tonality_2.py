
'''
Визначення тональності повідомлень стрічки новин з бібліотекою nltk та Українськими каркасами (словниками)
'''


import nltk
import time
import feedparser
import requests
import csv
import string
from nltk.corpus import stopwords
import pymorphy2
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#--------- завантаження сервісних атрибутів для бібліотеки nltk -----
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')


print('------------ парсінг стрічки новин -----------------------------------')
posts = []
rss_url = 'https://www.pravda.com.ua/ukr/rss/view_news/'
response = feedparser.parse(rss_url)
for each in response['entries']:
  if each['title'] in [x['title'] for x in posts]:
    pass
  else:
    posts.append({
        "title": each['title'],
        "link": each['links'][0]['href'],
        "tags": [x['term'] for x in each['tags']],
        "date": time.strftime('%Y-%m-%d', each['published_parsed'])
        })
for i, post in enumerate(p['title'] for p in posts):
  print(i, post)

print('\n')


print('------------ аналіз тональності стрічки новин ------------------------')
url = 'https://raw.githubusercontent.com/lang-uk/tone-dict-uk/master/tone-dict-uk.tsv'
r = requests.get(url)
with open(nltk.data.path[0]+'/tone-dict-uk.tsv', 'wb') as f:
    f.write(r.content)
d = {}
with open(nltk.data.path[0]+'/tone-dict-uk.tsv', 'r') as csv_file:
    for row in csv.reader(csv_file, delimiter='\t'):
        d[row[0]] = float(row[1])

SIA = SentimentIntensityAnalyzer()
SIA.lexicon.update(d)

for i, post in enumerate(p['title'] for p in posts):
  print(i, post, SIA.polarity_scores(post)["compound"])

print('\n')


print('------------ аналіз тональності стрічки новин БЕЗ стоп-слів --------------------')
url = 'https://raw.githubusercontent.com/olegdubetcky/Ukrainian-Stopwords/main/ukrainian'
r = requests.get(url)
with open(nltk.data.path[0]+'/corpora/stopwords/ukrainian', 'wb') as f:
    f.write(r.content)
# Retrieve HTTP meta-data
print(r.status_code)
print(r.headers['content-type'])
print(r.encoding)


stopwords = stopwords.words("ukrainian")

morph = pymorphy2.MorphAnalyzer(lang='uk')
stop_words = frozenset(stopwords + list(string.punctuation))
for i, post in enumerate(p['title'] for p in posts):
    sentences = nltk.sent_tokenize(post)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        without_stop_words = [word for word in words if not word in stop_words]
        normal_words = []
        for token in without_stop_words:
            p = morph.parse(token)[0]
            normal_words.append(p.normal_form)

        print(i, post, "RAW: ", SIA.polarity_scores(post)["compound"], "NORM: ",
              SIA.polarity_scores(' '.join(normal_words))["compound"])

