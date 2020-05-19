from google.colab import drive
drive.mount('/content/drive')

import nltk
import numpy as np
import pandas as pd
import csv
import re
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from unicodedata import normalize
import unicodedata

dataset = pd.read_csv ('/content/drive/My Drive/classificados.csv')

dataset.head()

dataset.info()

dataset.columns

dataset.groupby(by='Voto').size()

dataset.head(10)

dataset['Text'] = dataset.Text.str.lower()
dataset.head()

def remove_hashtags(item):
  if '#' in item:
    return item.replace('#', '').replace('#', '')
  else:
      return item


dataset['Text'] = [remove_hashtags(str(t)) for t in dataset['Text']]

dataset.head(10)

stopwords = ['pra', 'pro','to', 'ta','de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'a', 'com', 'nao', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'a', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'ha', 'nos', 'ja', 'esta', 'eu', 'tambam', 'sa3', 'pelo', 'pela', 'ata', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estao', 'vocaa', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'a s', 'minha', 'taam', 'numa', 'pelos', 'elas', 'havia', 'seja', 'qual', 'sera', 'na3s', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele', 'tu', 'te', 'vocaas', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 
'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'esta', 'estamos', 'estao', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estavamos', 'estavam', 'estivera', 'estivaramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivassemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'ha', 'havemos', 'hao', 'houve', 'houvemos', 'houveram', 'houvera', 'houvaramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvassemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houvera', 'houveremos', 'houverao', 'houveria', 'houveraamos', 'houveriam', 'sou', 'somos', 'sao', 'era', 'aramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fa ramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fa ssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'sera', 'seremos', 'serao', 'seria', 'seraamos', 'seriam', 'tenho', 'tem', 'temos', 'tam', 'tinha', 'tanhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivaramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivassemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'tera', 'teremos', 'terao', 'teria', 'teraamos', 'teriam']
print(len(stopwords))

def remove_stopWords(sentence):
    frase = []
    for word in sentence.split():
        if word not in stopwords:
            frase.append(word)
    return ' '.join(frase)

dataset['Text'] = [remove_stopWords(str(t)) for t in dataset['Text']]
dataset.head(10)

dataset['Text'] = dataset['Text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)

dataset.head(5000)

def removerAcentosECaracteresEspeciais(palavra):

    nfkd = unicodedata.normalize('NFKD', palavra)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    palavraSemAcento = re.sub('[^a-zA-Z0-9 \\\]', '', palavraSemAcento)

    return palavraSemAcento

dataset['Text'] = [removerAcentosECaracteresEspeciais(str(t)) for t in dataset['Text']]
dataset.head(5000)

dataset['Text'].isnull()

tweets = dataset['Text'].values.astype('U')
classes = dataset['Voto'].values.astype('U')

vectorizer = CountVectorizer(analyzer = 'word')
freq_tweets = vectorizer.fit_transform(tweets)

features = list(dataset.columns[6:7])
x = dataset[features]
print(features)

target = list(dataset.columns[7:8])
y = dataset[target]
print(target)

modelo = tree.DecisionTreeClassifier(criterion='gini')
modelo.fit(freq_tweets, classes)
#modelo = modelo.fit(x,y)

positivo = 0;
negativo = 0;
neutro = 0;
pos = []
neg = []
neu = []

for dado in dataset['Voto']:

  if dado == 1:
   positivo += 1
   pos.append(dado)

  if dado == -1:
    negativo +=1
    neg.append(dado)

  if dado == 0:
    neutro +=1
    neu.append(dado)

print(negativo)
#print(neg)

resultados = cross_val_predict(modelo, freq_tweets, classes, cv = 10)

metrics.accuracy_score(classes, resultados)
#metrics.accuracy_score(x,y)

sentimentos = ["Positivo", "Negativo", "Neutro"]

print(pd.crosstab(classes, resultados, rownames = ["Real"], colnames=["Predito"], margins=True))

print(metrics.classification_report(resultados, classes))