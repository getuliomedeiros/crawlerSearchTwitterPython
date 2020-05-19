import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import parse

from google.colab import drive
drive.mount('/content/drive')

fileWords = open("/content/drive/My Drive/word_keys.txt", "r") 
word_key = fileWords.readlines()

for i in word_key:

    i = i.rstrip()

    dataFrame = pd.read_csv(f'/content/drive/My Drive/CSV/{i}.csv')
    dataFrame.text.count()

    with open('/content/drive/My Drive/CSV/date.csv', mode='a', encoding='utf-8', newline='') as csv_file:
        fieldnames = ['tweet','quantidade']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        #writer.writeheader()
        writer.writerow({'tweet':f'{i}', 'quantidade': f'{dataFrame.text.count()}'})

cont = 0
for i in word_key:

    i = i.rstrip()

    dataFrameDatas = pd.read_csv(f'/content/drive/My Drive/CSV/{i}.csv')
    
    nova_data = parse(dataFrame.created_at.loc[cont])
    
    dataFrameDatas.strftime('%d/%m/%Y')

    dataFrame.created_at.count()

    with open('/content/drive/My Drive/IFPB/PROJETO/2019/Antigos/CSV/data_tweets.csv', mode='a', encoding='utf-8', newline='') as csv_file:
        fieldnames = ['tweet','quantidade','data']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        #writer.writeheader()
        writer.writerow({'data': f'{nova_data}', 'tweet':f'{i}', 'quantidade': f'{dataFrameDatas.text.count()}'})  
    cont += 1

dataFrame = pd.read_csv('/content/drive/My Drive/CSV/date.csv')
dataFrame

quantidade_dados = 0

dataFrame.quantidade

for i in dataFrame.quantidade:
    quantidade_dados += i

quantidade_dados

dataFrame.describe()

df = dataFrame.sort_values(by=['quantidade'])
dados = sns.barplot(df.tweet ,df.quantidade)
dados.set_xticklabels(dados.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
plt.title("Tweets Coletados")
plt.show()