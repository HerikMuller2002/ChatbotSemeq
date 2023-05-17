import os
import pandas as pd
import numpy as np

from services.preproccess import preprocess_lemma
from pickle import load
from keras.models import load_model
from random import choice


# retorna 0 ou 1 para cada palavra da bolsa de palavras
def bag_of_words(writing, words):
    # Pega as sentenças que são limpas e cria um pacote de palavras que são usadas para classes de previsão que são baseadas nos resultados que obtiver treinando o modelo.
    sentence_words = preprocess_lemma(writing).split()
    # cria uma matriz de N palavras
    bag = [0]*len(words)
    for setence in sentence_words:
        for i, word in enumerate(words):
            if word == setence:
                # atribui 1 no pacote de palavra se a palavra atual estiver na posição da frase
                bag[i] = 1
    return(np.array(bag))

# Faz a previsao do pacote de palavras, usa como limite de erro 0.25 para evitar overfitting, e classifica esses resultados por força da probabilidade.
def class_prediction(input_user, model_path, words_path, classes_path):
    model = load_model(model_path)
    words = load(open(words_path, 'rb'))
    classes = load(open(classes_path, 'rb'))
    # filtra as previsões abaixo de um limite 0.25
    prevision = bag_of_words(input_user, words)
    response_prediction = model.predict(np.array([prevision]))[0]
    results = [[index, response] for index, response in enumerate(response_prediction)]
    # verifica nas previsões se não há 1 na lista, se não há envia a resposta padrão (anything_else) ou se não corresponde a margem de erro
    if "1" not in str(prevision) or len(results) == 0 :
        results = [[0, response_prediction[0]]]
    # classifica por força de probabilidade
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


# pega a lista gerada, verifica e produz a maior parte das respostas com a maior probabilidade.
def get_response(intent, df_responses, df_intents):
    # Filtrando a linha do df_intents que corresponde à tag da intenção
    id_intent = df_intents.loc[df_intents['tag'] == intent[0]["intent"], 'id'].values[0]
    # Selecionando as linhas do df_responses que correspondem ao id_intent e guardando as respostas em uma lista
    responses = df_responses.loc[df_responses['intent_id'] == id_intent, 'response'].tolist()
    # Escolhendo aleatoriamente uma resposta da lista de respostas
    chosen_response = choice(responses)
    return chosen_response