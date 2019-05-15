import sys
import codecs
import random
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
import collections
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import nltk
import estnltk
from estnltk import Text
import numpy as np
import operator
from operator import itemgetter
import re


categories = ['1', '2', '3']
max_len = 0

def train(corp1, corp2, corp3):
    
    random.seed(42)
    #teksti sisselugemine
    corp1tekst = filetosents(corp1, categories.index('1'))
    corp2tekst = filetosents(corp2, categories.index('2'))
    corp3tekst = filetosents(corp3, categories.index('3'))
    #teksti sisselugemine
    corp_text = corp1tekst + corp2tekst + corp3tekst
    random.shuffle(corp_text)
    train_size = int(len(corp_text) * 0.8)
    
    X_train = [sent for sent, label in corp_text[:train_size]]
    y_train = to_categorical([label for sent, label in corp_text[:train_size]])
    X_val = [sent for sent, label in corp_text[train_size:]]
    y_val = to_categorical([label for sent, label in corp_text[train_size:]])
    
    # kodeeri tähthaaval
    tokenizer = Tokenizer(char_level=True) 
    # õpi tähestik selgeks meie treeningandmetel
    tokenizer.fit_on_texts(X_train)
    vocabulary_size = len(tokenizer.word_index) + 1
    X_train_vec = tokenizer.texts_to_sequences(X_train)
    X_val_vec = tokenizer.texts_to_sequences(X_val)
    
    X_train = [n for n, t in zip(X_train_vec, y_train) if not isinstance(n, float)]
    y_train = [t for n, t in zip(X_train_vec, y_train) if not isinstance(n, float)]
    X_val = [n for n, t in zip(X_val_vec, y_val) if not isinstance(n, float)]
    y_val = [t for n, t in zip(X_val_vec, y_val) if not isinstance(n, float)]

    lens = [len(v) for v in X_train if not isinstance(v, float)]
    max_len = max(lens)
    
    X_train_pad = sequence.pad_sequences(X_train, maxlen=max_len)
    X_val_pad = sequence.pad_sequences(X_val, maxlen=max_len)
    
    model = Sequential()
    model.add(Embedding(vocabulary_size, 10, input_length=max_len))
    model.add(LSTM(25, return_sequences=True))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(25, return_sequences=False))
    model.add(Dense(len(categories), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    try:
        model.fit(np.array(X_train_pad), np.array(y_train), epochs=5, batch_size=64, validation_data=(np.array(X_val_pad), np.array(y_val)))
    except KeyboardInterrupt:
        return model
    
    return model



def predict(model, sent):
    input_data = sequence.pad_sequences(tokenizer.texts_to_sequences([sent]), maxlen=max_len)
    
    # Teisendame väljundiks saadud numpy float32 array tavaliseks pythoni listiks ujukomaarvudest
    prediction = list(float(v) for v in model.predict(input_data)[0])
    # Valime suurima väljundi, see on kõige tõenäolisema kategooria hinnatud tõenäosus
    confidence = max(prediction)
    # Tagastame kategooria numbri (hinnangud on samas järjekorras, kui kategooriad), ja kõigi kategooriate 'kindluse'
    return (prediction.index(confidence), prediction)
    
    
def filetosents(file, catindex):
    corp_file = codecs.open(file, 'r', encoding='utf-8')
    sents = list(filter(lambda n: n is not None and n.strip() is not "" and n.strip() is not '', corp_file.read().split("\n")))
    corp_text = []
    for sent in sents:
        for s in Text(sent).split_by('sentences'):
            a = [';',',','.','-','!']
            if any(x in s['text'] for x in a):
                for cs in re.split(',|-|;|.|!',s['text']): corp_text.append(cs.strip())
            else: corp_text.append(s['text'].strip())
    corp_text = [(sent.strip(), catindex) for sent in corp_text if len(sent) is not 0]
    return corp_text
    
    
if __name__ == '__main__':
    
    corp1 = "janeausten.txt"
    corp2 = "poe.txt"
    corp3 = "shakespeare.txt"
    
    random.seed(42)
    #teksti sisselugemine
    corp1tekst = filetosents(corp1, categories.index('1'))
    corp2tekst = filetosents(corp2, categories.index('2'))
    corp3tekst = filetosents(corp3, categories.index('3'))
    #teksti sisselugemine
    corp_text = corp1tekst + corp2tekst + corp3tekst
    random.shuffle(corp_text)
    corp_text = corp_text[:50000]
    train_size = int(len(corp_text) * 0.8)
    train_sents = corp_text[:train_size]
    
    test_sents = corp_text[train_size:]
    train_size = int(len(train_sents) * 0.8)
    X_train = [sent for sent, label in train_sents[:train_size]]
    y_train = [label for sent, label in train_sents[:train_size]]
    X_val = [sent for sent, label in train_sents[train_size:]]
    y_val = [label for sent, label in train_sents[train_size:]]
    
    # kodeeri tähthaaval
    tokenizer = Tokenizer(char_level=True) 
    # õpi tähestik selgeks meie treeningandmetel
    tokenizer.fit_on_texts(X_train)
    vocabulary_size = len(tokenizer.word_index) + 1
    X_train = [sentwork(sent) for sent in X_train]
    X_val = [sentwork(sent) for sent in X_val]
    max_len = max(len(v) for v in X_train if not isinstance(v, float))
    
    model = Sequential()
    model.add(Embedding(vocabulary_size, 10, input_length=200))
    model.add(LSTM(25, return_sequences=True))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(25, return_sequences=False))
    model.add(Dense(len(categories), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

    #model = train(corp1, corp2, corp3)
    eval = []
    for sent, label in test_sents:
        cat = predict(model, sent)[0]
        if cat == label: eval.append(1)
        else: eval.append(0)
    print(sum(eval)/len(eval) *100, "%")
    