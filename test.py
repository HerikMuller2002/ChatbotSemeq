import os
import json
from model.log import Log_chat
from model.extract import class_prediction, get_response
from keras.models import load_model
from model.preprocess import Tratamento
from model.preprocess import Correlacao
from numpy import delete
from sklearn.feature_extraction.text import TfidfVectorizer # pip install scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
from random import choice

def chatbot_run(input_user):
    input_user = Tratamento.preprocess_input(input_user)
    id_response = 0
    # extrai o modelo usando o keras
    # model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Chatbot\\model\\model.h5"))
    model_path = "./model/model.h5"
    model = load_model(model_path)
    # abre o banco de dados para carregar as intenções
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ChatbotSemeq\\database\\intents.json"))
    with open(db_path,'r',encoding="UTF-8") as banco:
        intents_db = json.load(banco)

    # armazena a lista de palavras censuradas
    json_censored_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ChatbotSemeq\\database\\censored.json"))
    with open(json_censored_path,'r',encoding="UTF-8") as banco:
        intents = json.load(banco)
    censored,id = Correlacao.tf_idf(input_user,intents["intents"][0]["patterns"])
    if censored > 0.1:
        list_response = intents["intents"][0]["responses"]
        response = choice(list_response)
        print("não vai passar")

a = chatbot_run("vai se fuder")
print()
b = chatbot_run("olá")