import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

#fileWords = open("word_keys.txt", "r") 
#word_key = fileWords.readlines()

from subprocess import check_output

#for i in word_key:

dataset = pd.read_csv('../database/arquivo.csv', encoding='utf-8')
dataset.count()

dataset[dataset.Voto == 0].count()
dataset[dataset.Voto == 1].count()
dataset[dataset.Voto == -1].count()
dataset.head()

tweets = dataset["Text"].values.astype('U')
classes = dataset["Voto"].values.astype('U')

vectorizer = CountVectorizer(analyzer = "word")
freq_tweets = vectorizer.fit_transform(tweets)

modelo = MultinomialNB()
modelo.fit(freq_tweets, classes)

resultados = cross_val_predict(modelo, freq_tweets, classes, cv = 10)

metrics.accuracy_score(classes, resultados)

sentimentos = ["Positivo", "Negativo", "Neutro"]
print(metrics.classification_report(classes, resultados, sentimentos))

print(pd.crosstab(classes, resultados, rownames = ["Real"], colnames=["Predito"], margins=True))

vectorizer = CountVectorizer(ngram_range = (1, 2))
freq_tweets = vectorizer.fit_transform(tweets)

modelo = MultinomialNB()
modelo.fit(freq_tweets, classes)

resultados = cross_val_predict(modelo, freq_tweets, classes, cv = 10)

metrics.accuracy_score(classes, resultados)

print(metrics.classification_report(classes, resultados, sentimentos))

print(pd.crosstab(classes, resultados, rownames = ["Real"], colnames = ["Predito"], margins = True))
