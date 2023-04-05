import pickle
import random
import numpy as np
import os
from preprocess import preprocess_model
from preprocess import preprocess_lemma
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


####################################################################
from database import read_db
path_models = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
path_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))

df_intents = read_db('intents')
df_patterns = read_db('patterns')


######################################################################
def train_model(df_intents,df_patterns,path):

    classes = [j['tag'] for i,j in df_intents.iterrows()]
    words,documents = preprocess_model(df_patterns,df_intents)

    # classificamos nossas listas
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    # salvamos as palavras e classes nos arquivos pkl
    words_path = os.path.join(path, "words.pkl")
    classes_path = os.path.join(path, "classes.pkl")

    pickle.dump(words,open(words_path, 'wb'))
    pickle.dump(classes,open(classes_path, 'wb'))

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
        pattern_words = [preprocess_lemma(word.lower()) for word in pattern_words]

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

    model_path = os.path.join(path, "model.h5")
    model.save(model_path, m)