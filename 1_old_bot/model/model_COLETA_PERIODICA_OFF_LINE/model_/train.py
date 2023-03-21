import json
import pickle
import nltk
import random
import numpy as np
import os
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
# abrindo o "banco de dados" temporario em json
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))
json_path = os.path.join(db_path, "intents.json")
with open(json_path,'r',encoding="UTF-8") as banco:
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

# salvamos as palavras e classes nos arquivos pkl
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
words_path = os.path.join(models_path, "words.pkl")
classes_path = os.path.join(models_path, "classes.pkl")

pickle.dump(words,open(words_path, 'wb'))
pickle.dump(classes,open(classes_path, 'wb'))

#############################################################################

# inicializamos o treinamento
training = []
output_empty = [0] * len(classes)
for document in documents:
    # inicializamos o saco de palavras 
    bag = []

    # listamos as palavras do pattern
    pattern_words = document[0]

    # lematizamos cada palavra 
    # na tentativa de representar palavras relacionadas
    pattern_words = [lemmatizer.lemmatize( word.lower()) for word in pattern_words]

    # criamos nosso conjunto de palavras com 1, 
    # se a correspondência de palavras for encontrada no padrão atual
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # adicionamos zeros à lista para preencher com o tamanho máximo
    # de palavras no vocabulário
    while len(bag) < len(words):
        bag.append(0)

    # output_row atuará como uma chave para a lista, 
    # onde a saida será 0 para cada tag e 1 para a tag atual
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

# embaralhamos nosso conjunto de treinamentos e transformamos em numpy array
random.shuffle(training)
training = np.array(training, dtype=object)
# criamos lista de treino sendo x os patterns e y as intenções
x = list(training[:, 0])
y = list(training[:, 1])

#############################################################################

# Criamos nosso modelo com 3 camadas. 
# Primeira camada de 128 neurônios, 
# segunda camada de 64 neurônios e terceira camada de saída 
# contém número de neurônios igual ao número de intenções para prever a intenção de saída com softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# O modelo é compilado com descida de gradiente estocástica 
# com gradiente acelerado de Nesterov.
# A ideia da otimização do Momentum de Nesterov, ou Nesterov Accelerated Gradient (NAG), 
# é medir o gradiente da função de custo não na posição local,
# mas ligeiramente à frente na direção do momentum. 
# A única diferença entre a otimização de Momentum é que o gradiente é medido em θ + βm em vez de em θ.
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

# ajustamos e salvamos o modelo
m = model.fit(np.array(x), np.array(y), epochs=200, batch_size=5, verbose=1)

models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
model_path = os.path.join(models_path, "model.h5")

model.save(model_path, m)