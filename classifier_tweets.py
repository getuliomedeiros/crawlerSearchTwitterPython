from nltk import word_tokenize
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

fileWords = open("word_keys.txt", "r") 
word_key = fileWords.readlines()

for i in word_key:
  dataset = pd.read_csv(f'{i}.csv', encoding='utf-8')
  
  dataset.Text.head(50)
  
  dataset[dataset.Classificacao=='Neutro'].count()
  dataset[dataset.Classificacao=='Positivo'].count()
  dataset[dataset.Classificacao=='Negativo'].count()
  
  dataset.Classificacao.value_counts().plot(kind='bar')
  dataset.count()

  dataset.drop_duplicates(['Text'], inplace=True)
  dataset.Text.count()

  tweets = dataset['Text']
  classes = dataset['Classificacao']

  import nltk
  nltk.download('stopwords')
  nltk.download('rslp')

  def RemoveStopWords(instancia):
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))
  
  def Stemming(instancia):
    stemmer = nltk.stem.RSLPStemmer()
    palavras = []
    for w in instancia.split():
        palavras.append(stemmer.stem(w))
    return (" ".join(palavras))
  
  def Limpeza_dados(instancia):
    # remove links, pontos, virgulas,ponto e virgulas dos tweets
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    return (instancia)
  
  RemoveStopWords('Eu não gosto do partido, e também não votaria novamente nesse governante!')
  Stemming('Eu não gosto do partido, e também não votaria novamente nesse governante!')
  Limpeza_dados('Assita aqui o video do Governador falando sobre a CEMIG https://www.uol.com.br :) ;)')

  def Preprocessing(instancia):
    stemmer = nltk.stem.RSLPStemmer()
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [stemmer.stem(i) for i in instancia.split() if not i in stopwords]
    return (" ".join(palavras))

  # Aplica a função em todos os dados:
  tweets = [Preprocessing(i) for i in tweets]

  Preprocessing('Eu não gosto do partido, e também não votaria novamente nesse governante. Assita o video aqui https:// :)')

  tweets[:50]

  vectorizer = CountVectorizer(analyzer="word")

  freq_tweets = vectorizer.fit_transform(tweets)
  type(freq_tweets)  

  modelo = MultinomialNB()
  modelo.fit(freq_tweets,classes)

  freq_tweets.shape
  freq_tweets.A

  # defina instâncias de teste dentro de uma lista
  testes = ['Esse governo está no início, vamos ver o que vai dar',
          'Estou muito feliz com o governo de Minas esse ano',
          'O estado de Minas Gerais decretou calamidade financeira!!!',
          'A segurança desse país está deixando a desejar',
          'O governador de Minas é mais uma vez do PT']

  testes = [Preprocessing(i) for i in testes]


  # Transforma os dados de teste em vetores de palavras.
  freq_testes = vectorizer.transform(testes)


  # Fazendo a classificação com o modelo treinado.
  for t, c in zip (testes,modelo.predict(freq_testes)):
      print (t +", "+ c)

  # Probabilidades de cada classe
  print (modelo.classes_)
  modelo.predict_proba(freq_testes).round(2)


  def marque_negacao(texto):
      negacoes = ['não','not']
      negacao_detectada = False
      resultado = []
      palavras = texto.split()
      for p in palavras:
          p = p.lower()
          if negacao_detectada == True:
              p = p + '_NEG'
          if p in negacoes:
              negacao_detectada = True
          resultado.append(p)
      return (" ".join(resultado))

  marque_negacao('Eu gosto do partido, votaria novamente nesse governante!')
  marque_negacao('Eu Não gosto do partido, e também não votaria novamente nesse governante!')

  from sklearn.pipeline import Pipeline

  pipeline_simples = Pipeline([
    ('counts', CountVectorizer()),
    ('classifier', MultinomialNB())
  ])


  pipeline_negacoes = Pipeline([
    ('counts', CountVectorizer(tokenizer=lambda text: marque_negacao(text))),
    ('classifier', MultinomialNB())
  ])


  pipeline_simples.fit(tweets,classes)
  pipeline_simples.steps

  pipeline_negacoes.fit(tweets,classes)

  pipeline_negacoes.steps

  resultados = cross_val_predict(pipeline_simples, tweets, classes, cv=10)
  metrics.accuracy_score(classes,resultados)


  sentimento=['Positivo','Negativo','Neutro']
  print (metrics.classification_report(classes,resultados,sentimento))

  print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True)

  resultados = cross_val_predict(pipeline_negacoes, tweets, classes, cv=10)

  metrics.accuracy_score(classes,resultados)

  sentimento=['Positivo','Negativo','Neutro']
  print (metrics.classification_report(classes,resultados,sentimento))

  print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True)

  'eu gosto', 'gosto do' , 'do brasil'

  vectorizer = CountVectorizer(ngram_range=(1,2))
  freq_tweets = vectorizer.fit_transform(tweets)
  modelo = MultinomialNB()
  modelo.fit(freq_tweets,classes)

  resultados = cross_val_predict(modelo, freq_tweets, classes, cv=10)
  metrics.accuracy_score(classes,resultados)

  sentimento=['Positivo','Negativo','Neutro']
  print (metrics.classification_report(classes,resultados,sentimento))

  print (pd.crosstab(classes, resultados, rownames=['Real'], colnames=['Predito'], margins=True)