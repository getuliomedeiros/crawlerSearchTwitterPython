import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import re
from datetime import datetime
from dateutil.parser import parse
from unicodedata import normalize
import unicodedata

from google.colab import drive
drive.mount('/content/drive')

fileWords = open("/content/drive/My Drive/word_keys.txt", "r") 
word_key = fileWords.readlines()

stopwords = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'a', 'com', 'nao', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'a', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'ha', 'nos', 'ja', 'esta', 'eu', 'tambam', 'sa3', 'pelo', 'pela', 'ata', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estao', 'vocaa', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'a s', 'minha', 'taam', 'numa', 'pelos', 'elas', 'havia', 'seja', 'qual', 'sera', 'na3s', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele', 'tu', 'te', 'vocaas', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 
'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'esta', 'estamos', 'estao', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estavamos', 'estavam', 'estivera', 'estivaramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivassemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'ha', 'havemos', 'hao', 'houve', 'houvemos', 'houveram', 'houvera', 'houvaramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvassemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houvera', 'houveremos', 'houverao', 'houveria', 'houveraamos', 'houveriam', 'sou', 'somos', 'sao', 'era', 'aramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fa ramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fa ssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'sera', 'seremos', 'serao', 'seria', 'seraamos', 'seriam', 'tenho', 'tem', 'temos', 'tam', 'tinha', 'tanhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivaramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivassemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'tera', 'teremos', 'terao', 'teria', 'teraamos', 'teriam']
print(len(stopwords))

def remove_stopWords(sentence):
    frase = []
    for word in sentence.split():
        if word not in stopwords:
            frase.append(word)
    return ' '.join(frase)

with open('/content/drive/My Drive/CSV/fileTexts.csv', mode='w', encoding='utf-8', newline='') as csv_file:
          fieldnames = ['text']
          writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
          writer.writeheader()

def removerAcentosECaracteresEspeciais(palavra):

    nfkd = unicodedata.normalize('NFKD', palavra)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])

    return re.sub('[^a-zA-Z0-9 \\\]', '', palavraSemAcento)

for i in word_key:

    i = i.rstrip()

    dataFrame = pd.read_csv(f'/content/drive/My Drive/CSV/{i}.csv')
    dataset = dataFrame['text']

    for j in dataset:
      meuTweet = ""
      target = normalize('NFKD', j).encode('ASCII','ignore').decode('ASCII').lower().rstrip()
      meuTweet += " " + target.rstrip() 
      tweetProcessado = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', meuTweet)
      tweetProcessado = removerAcentosECaracteresEspeciais(tweetProcessado)
      newTweet = remove_stopWords(tweetProcessado)
    
      with open('/content/drive/My Drive/CSV/fileTexts.csv', mode='a', encoding='utf-8', newline='') as csv_file:
          fieldnames = ['text']
          writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      
          writer.writerow({'text':f'{newTweet}'})

cont = 0

dataFrame = pd.read_csv('/content/drive/My Drive/CSV/fileTexts.csv')
dataFrame

