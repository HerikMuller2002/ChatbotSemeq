import json
import pickle
import nltk
import random
import numpy as np
import os
import re
import spacy

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

######################################################################

# inicializaremos nossa lista de palavras, classes, documentos e 
# definimos quais palavras serão ignoradas
words = []
documents = []
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Chatbot\\database"))
json_path = os.path.join(db_path, "intents.json")
with open(json_path,'r',encoding="UTF-8") as banco:
    intents = json.load(banco)
# adicionamos as tags em nossa lista de classes
classes = [i['tag'] for i in intents['intents']]

# inicializaremos nossa lista de palavras, classes, documentos e 
# definimos quais palavras serão ignoradas
words = []
documents = []
# adicionamos as tags em nossa lista de classes
classes = [i['tag'] for i in intents['intents']]
ignore_words = ["!", "@", "#", "$", "%", "*", "?"]

# percorremos nosso array de objetos
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # com ajuda no nltk fazemos aqui a tokenizaçao dos patterns 
        # e adicionamos na lista de palavras
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        # adiciona aos documentos para identificarmos a tag para a mesma
        documents.append((word, intent['tag']))

# # classificamos nossas listas
# words = sorted(list(set(words)))
# classes = sorted(list(set(classes)))

nlp = spacy.load("pt_core_news_sm")

def preprocess(text,lematizar=True):
    text = text.lower()
    if lematizar:
        # encontrar radical das palavras (lematização)
        documento = nlp(text)
        text = []
        for token in documento:
            text.append(token.lemma_)
        text = ' '.join([str(elemento) for elemento in text if not elemento.isdigit()])
    # tirar pontuações, acentos e espaços extras
    text = re.sub(r"[!#$%&'()*+,-./:;<=>?@[^_`{|}~]+", ' ', re.sub('[áàãâä]', 'a', re.sub('[éèêë]', 'e', re.sub('[íìîï]', 'i', re.sub('[óòõôö]', 'o', re.sub('[úùûü]', 'u', text))))))
    # tirar espaços em branco
    text = re.sub(r'\s+', ' ',text)
    return text.strip()

words = [preprocess(w) for w in words if w not in ignore_words]
print(words)


# percorremos nosso array de objetos
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # com ajuda no nltk fazemos aqui a tokenizaçao dos patterns 
        # e adicionamos na lista de palavras
        word = nltk.word_tokenize(pattern)
        words.extend(word)

        # adiciona aos documentos para identificarmos a tag para a mesma
        documents.append((word, intent['tag']))

# lematizamos as palavras ignorando os palavras da lista ignore_words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
print(words)