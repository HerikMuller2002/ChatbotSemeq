import json
import nltk
import os
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

######################################################################

# inicializaremos nossa lista de palavras, classes, documentos e 
# definimos quais palavras serão ignoradas
words = []
documents = []

with open("database\\intents.json",'r',encoding="UTF-8") as banco:
    intents = json.load(banco)
    
################################################################################################################
#--------------------------------------------------------------------------------------------------------------#
################################################################################################################

# abrir a tabela de intents do banco de dados
# tokenizar (matriz) todos os pattern (possíveis perguntas do usuário) e armazenar em words (lista)
# adicionamos cada token do patern com a sua "tag" (intenção) em documents (lista)
# exmplo:
    # word = nltk.word_tokenize(pattern)
    # words.extend(word)
    # documents.append((word, intent['tag']))

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

# lematizamos as palavras ignorando os palavras da lista ignore_words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]

################################################################################################################
#--------------------------------------------------------------------------------------------------------------#
################################################################################################################

# classificamos nossas listas
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))