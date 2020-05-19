import csv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import classification_report
from unicodedata import normalize
import unicodedata
import re

vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type =  'post'
oov_tok = "<OOV>"
training_portion = .8

sentences = []
labels = []
stopwords = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'a', 'com', 'nao', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'a', 'seu', 'sua', 'ou', 'ser', 'quando', 'muito', 'ha', 'nos', 'ja', 'esta', 'eu', 'tambam', 'sa3', 'pelo', 'pela', 'ata', 'isso', 'ela', 'entre', 'era', 'depois', 'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estao', 'vocaa', 'tinha', 'foram', 'essa', 'num', 'nem', 'suas', 'meu', 'a s', 'minha', 'taam', 'numa', 'pelos', 'elas', 'havia', 'seja', 'qual', 'sera', 'na3s', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fosse', 'dele', 'tu', 'te', 'vocaas', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 
'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'esta', 'estamos', 'estao', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estavamos', 'estavam', 'estivera', 'estivaramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivassemos', 'estivessem', 'estiver', 'estivermos', 'estiverem', 'hei', 'ha', 'havemos', 'hao', 'houve', 'houvemos', 'houveram', 'houvera', 'houvaramos', 'haja', 'hajamos', 'hajam', 'houvesse', 'houvassemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houvera', 'houveremos', 'houverao', 'houveria', 'houveraamos', 'houveriam', 'sou', 'somos', 'sao', 'era', 'aramos', 'eram', 'fui', 'foi', 'fomos', 'foram', 'fora', 'fa ramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fa ssemos', 'fossem', 'for', 'formos', 'forem', 'serei', 'sera', 'seremos', 'serao', 'seria', 'seraamos', 'seriam', 'tenho', 'tem', 'temos', 'tam', 'tinha', 'tanhamos', 'tinham', 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivaramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivassemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'tera', 'teremos', 'terao', 'teria', 'teraamos', 'teriam']

def removerAcentosECaracteresEspeciais(palavra):

    nfkd = unicodedata.normalize('NFKD', palavra)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    palavraSemAcento = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', palavraSemAcento)
    palavraSemAcento = normalize('NFKD', palavraSemAcento).encode('ASCII','ignore').decode('ASCII').lower().rstrip()

    return re.sub('[^a-zA-Z0-9 \\\]', '', palavraSemAcento)

with open('/content/drive/My Drive/IFPB/PROJETO/2020/classificados.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[6])
        sentence = row[7]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
            sentence = removerAcentosECaracteresEspeciais(sentence)
        sentences.append(sentence)

    
print(len(labels))
print(len(sentences))
print(labels[0])
print(sentences[0])

train_size = (int) (training_portion * len(labels))

train_sentences = sentences[0:train_size]
train_labels = labels[0:train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences,maxlen=max_length, padding=padding_type)

print(len(train_sequences[0]))
print(len(train_padded[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))

validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences,maxlen=max_length, padding=padding_type)

print(len(validation_sequences))
print(validation_padded.shape)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = pad_sequences(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = pad_sequences(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='sigmoid')

])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq))
