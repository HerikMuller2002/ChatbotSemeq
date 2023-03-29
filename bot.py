import os
import json
import pandas as pd
import numpy as np
from extract import class_prediction, get_response
from keras.models import load_model
from preprocess import Tratamento
from preprocess import Correlacao
from random import choice
from models.modelos_intencoes.modelo_suporte.get_intent import get_subject
from models.modelos_intencoes.modelo_suporte.filter import get_solution
# from models.modelos_intencoes.modelo_suporte.get_intent import get_get_response_question
from logs import log_chat

def chatbot_run(input_user):
    if input_user == "clear":
        log_chat.clear_log()
    input_user = Tratamento.preprocess_input(input_user)
    with open("models\\modelos_intencoes\\censored\\intents.json",'r',encoding="UTF-8") as bd:
        list_censored = json.load(bd)
    censored,id = Correlacao.tf_idf(input_user,list_censored["intents"][0]["patterns"])
    if censored > 0.2:
        list_response = list_censored["intents"][0]["responses"]
        response = choice(list_response)
    else:
        with open("models\\modelo_contexto\\intents.json",'r',encoding="UTF-8") as bd:
            list_context = json.load(bd)
        model_path = "models\\modelo_contexto\\model.h5"
        words_path = "models\\modelo_contexto\\words.pkl"
        classes_path = "models\\modelo_contexto\\classes.pkl"

        context_user = class_prediction(input_user, model_path,words_path,classes_path)
        context = get_response(context_user, list_context)

        if context == "anything_else":
            with open("models\\modelos_intencoes\\anything_else\\intents.json",'r',encoding="UTF-8") as bd:
                list_intents = json.load(bd)
            intent_user = [{"intent": "anything_else", "probability": 1.0}]
            response = get_response(intent_user, list_intents)
        elif context == "casual":
            with open("models\\modelos_intencoes\\modelo_casual\\intents.json",'r',encoding="UTF-8") as bd:
                list_intents = json.load(bd)
            model_path = "models\\modelos_intencoes\\modelo_casual\\model.h5"
            words_path = "models\\modelos_intencoes\\modelo_casual\\words.pkl"
            classes_path = "models\\modelos_intencoes\\modelo_casual\\classes.pkl"
            intent_user = class_prediction(input_user, model_path,words_path,classes_path)
            if intent_user[0]['intent'] == 'bye':
                log_chat.clear_log()
            response = get_response(intent_user, list_intents)
        elif context == "casual":
            with open("models\\modelos_intencoes\\modelo_casual\\intents.json",'r',encoding="UTF-8") as bd:
                list_intents = json.load(bd)
            model_path = "models\\modelos_intencoes\\modelo_casual\\model.h5"
            words_path = "models\\modelos_intencoes\\modelo_casual\\words.pkl"
            classes_path = "models\\modelos_intencoes\\modelo_casual\\classes.pkl"
            intent_user = class_prediction(input_user, model_path,words_path,classes_path)
            if intent_user[0]['intent'] == 'bye':
                log_chat.clear_log()
            response = get_response(intent_user, list_intents)
        else:
            json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
            if os.path.exists(json_path):
                if os.path.isfile(os.path.join('logs\\log.json')):
                    with open((json_path,'log.json'), 'r', encoding='utf-8') as f:
                        log = json.load(f)
            subject,device,interface,model,problem,response = get_solution(input_user)
            log_chat.log_chat(input_user,context,response,subject,device,interface,model,problem)

    response_bot = []
    if type(response) == list:
        for i in response:
            response_bot.append({"text":i})
    else:
        response_bot.append({"text":response})

    return response_bot