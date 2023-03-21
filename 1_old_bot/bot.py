import os
import json
import pandas as pd
import numpy as np
from model.log import Log_chat
from model.extract import class_prediction, get_response
from keras.models import load_model
from model.preprocess import Tratamento
from model.preprocess import Correlacao
from random import choice

def chatbot_run(input_user):
    input_user = Tratamento.preprocess_input(input_user)
    id_response = 0
    # extrai o modelo usando o keras
    # model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Chatbot\\model\\model.h5"))
    # abre o banco de dados para carregar as intenções
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ChatbotSemeq\\database\\intents.json"))
    with open(db_path,'r',encoding="UTF-8") as banco:
        intents_db = json.load(banco)
    
    # armazena a lista de palavras censuradas
    json_censored_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ChatbotSemeq\\database\\censored.json"))
    with open(json_censored_path,'r',encoding="UTF-8") as banco:
        intents = json.load(banco)
    censored,id = Correlacao.tf_idf(input_user,intents["intents"][0]["patterns"])
    if censored > 0.2:
        list_response = intents["intents"][0]["responses"]
        response = choice(list_response)
    else:
        model_path = "./model/model.h5"
        model = load_model(model_path)
        intent_user = class_prediction(input_user, model)
        if intent_user[0]['intent'] == 'problem_solution':
            table_problemas = pd.read_csv("database\\tabela_problemas.csv", encoding="iso-8859-1")
            vetor,id = Correlacao.tf_idf(input_user,table_problemas['problema'].tolist())
            if vetor > 0.3:
                response = table_problemas['problema'][id]
            else:
                response = get_response([{'intent': 'welcome_problem', 'probability': '1.0'}], intents_db)
        else:
        # # carrega a intenção do log da última conversa
        # intent_log = ...
        # # verifica se a intenção era um feedback
        # if ...:
        #     ...
        # # verifica se a intenção era uma pergunta filtro para o usuário responder 
        # elif ...:
        #     ...
        # # se não for nenhuma das intenções de resposta acima, chama o modelo para interagir
        # else:
            response = get_response(intent_user, intents_db)
    #     list_response = get_response(intent_user, intents_db)
    #     response = [id_response]
    # # cria um log da conversa
    # Log_chat(intent_user,input_user,response,list_response=0,id_response=0)
    return response

a = chatbot_run('olá')