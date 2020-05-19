from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
import csv
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from string import punctuation
import unicodedata
import re

from google.colab import drive
drive.mount('/content/drive')

dados_textos = pd.read_csv('/content/drive/My Drive/classificados.csv')

dados_textos.head()

def limpeza_dados(tuites, text_field):
    tuites[text_field] = tuites[text_field].str.lower()
    tuites[text_field] = tuites[text_field].str.replace(r"#", " ") #remove hashtags
    tuites[text_field] = tuites[text_field].str.replace(r"http", " ")
    tuites[text_field] = tuites[text_field].str.replace(r"http\S+", " ")
    tuites[text_field] = tuites[text_field].str.replace(r"@", "at")
    tuites[text_field] = tuites[text_field].str.replace(r"\n", " ") #remove as linhas em branco
    return tuites

X = dados_textos["Text"].values.astype('U')
y = dados_textos["Voto"].values.astype('U')

#limpeza dos dados
dados_textos_limpos = limpeza_dados(dados_textos, "Text")

stopwords = ['pra', 'pro','to', 'ta','de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'a', 'com', 'nao', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'a', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'ha', 'nos', 'ja', 'esta', 'eu', 'tambam', 'sa3', 'pelo', 'pela', 'ata', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estao', 'vocaa', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'a s', 'minha', 'taam', 'numa', 'pelos', 'elas', 'havia', 'seja', 'qual', 'sera', 'na3s', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele', 'tu', 'te', 'vocaas', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 
'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'esta', 'estamos', 'estao', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estavamos', 'estavam', 'estivera', 'estivaramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivassemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'ha', 'havemos', 'hao', 'houve', 'houvemos', 'houveram', 'houvera', 'houvaramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvassemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houvera', 'houveremos', 'houverao', 'houveria', 'houveraamos', 'houveriam', 'sou', 'somos', 'sao', 'era', 'aramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fa ramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fa ssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'sera', 'seremos', 'serao', 'seria', 'seraamos', 'seriam', 'tenho', 'tem', 'temos', 'tam', 'tinha', 'tanhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivaramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivassemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'tera', 'teremos', 'terao', 'teria', 'teraamos', 'teriam']
print(len(stopwords))

def remove_stopWords(sentence):
    frase = []
    for word in sentence.split():
        if word not in stopwords:
           # semStop = [p for p in word.split() if p not in stopwords]
            frase.append(word)
    return ' '.join(frase)

dados_textos['Text'] = [remove_stopWords(str(t)) for t in dados_textos['Text']]

def removerAcentosECaracteresEspeciais(palavra):

    nfkd = unicodedata.normalize('NFKD', palavra)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    palavraSemAcento = re.sub('[^a-zA-Z0-9 \\\]', '', palavraSemAcento)

    return palavraSemAcento

dados_textos['Text'] = [removerAcentosECaracteresEspeciais(str(t)) for t in dados_textos['Text']]

#regravação do arquivo com os dados limpos 
dados_textos_limpos.to_csv("dados_limpos.csv")

dados_textos_limpos.head()

dados_textos_limpos.tail()

vectorizer = CountVectorizer(analyzer = "word")
X_vetor = vectorizer.fit_transform(X)
X_vetor.shape

X_train, X_test, y_train, y_test = train_test_split(X_vetor, y, test_size = 0.3)

C = 1.0  # SVM regularization parameter
classificador = svm.SVC(kernel='linear', C=C)

#treinando o modelo
classificador.fit(X_train, y_train)

classificador.predict(X_test)

y_pred = classificador.predict(X_test)

classificador.score(X_test,y_test)

print(classification_report(y_test, y_pred))

import pickle

#Importação dos tweets
dataFrame = pd.read_csv('/content/drive/My Drive/CSV/fileTexts.csv')
Z = dataFrame['text'].values.astype('U')
Z

#Vetorizando tweets
Z_vetor = vectorizer.transform(Z)
Z_vetor.shape

classificador.predict(Z_vetor)

dataPredict = classificador.predict(Z_vetor)

with open('/content/drive/My Drive/dataPredict.csv', mode='w', encoding='utf-8', newline='') as csv_file:
          fieldnames = ['predicao']
          writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
          writer.writeheader()

for i in dataPredict:
  with open('/content/drive/My Drive/dataPredict.csv', mode='a', encoding='utf-8', newline='') as csv_file:
          fieldnames = ['predicao']
          writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
          writer.writerow({'predicao':f'{i}'})

#Persistencia do modelo em arquivo .sav
pickle.dump(classificador, open('/content/drive/My Drive/modelos/AllDataSVM.sav', 'wb'))

#Chamada do modelo salvo e previsão
modelo = pickle.load(open('/content/drive/My Drive/modelos/AllDataSVM.sav', 'rb'))
modelo.predict(Z_vetor)