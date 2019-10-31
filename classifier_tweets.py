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

dataset = pd.read_csv('./tagsClassificadas.csv', encoding='utf-8')
dataset.count()

dataset[dataset.Voto == 0].count()

dataset[dataset.Voto == 1].count()

dataset[dataset.Voto == -1].count()

dataset.head()

tweets = dataset["text"].values.astype('U')
tweets

classes = dataset["Voto"].values.astype('U')
classes

vectorizer = CountVectorizer(analyzer = "word")
freq_tweets = vectorizer.fit_transform(tweets)

modelo = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
modelo.fit(freq_tweets, classes)

testes = ["Esse governo está no início, vamos ver o que vai dar",
          "Polícia só não é bem-vinda onde acontecem crimes (estudantes reclamam por sofrer estupro em universidades);",
          "O estado de Minas Gerais decretou calamidade financeira!!!",
          "A segurança desse país está deixando a desejar",
          "É um desgoverno literalmente contra a educação!! Vergonhoso!! ",
          "Tbm com a presença do PT, PSOL, MST E SINDICATOS... FALTOU SÓ OS AMIGOS DAS FACÇÕES."]

freq_testes = vectorizer.transform(testes)
modelo.predict(freq_testes)

resultados = cross_val_predict(modelo, freq_tweets, classes, cv = 10)
resultados

metrics.accuracy_score(classes, resultados)

sentimentos = ["Positivo", "Negativo", "Neutro"]
print(metrics.classification_report(classes, resultados, sentimentos))

print(pd.crosstab(classes, resultados, rownames = ["Real"], colnames=["Predito"], margins=True))

vectorizer = CountVectorizer(ngram_range = (1, 2))
freq_tweets = vectorizer.fit_transform(tweets)

modelo = MultinomialNB(alpha=2.3, class_prior=None, fit_prior=True)
modelo.fit(freq_tweets, classes)

resultados = cross_val_predict(modelo, freq_tweets, classes, cv = 10)
resultados

metrics.accuracy_score(classes, resultados)

print(metrics.classification_report(classes, resultados, sentimentos))

print(pd.crosstab(classes, resultados, rownames = ["Real"], colnames = ["Predito"], margins = True))
