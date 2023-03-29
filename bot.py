import json
import pandas as pd
import os
from extract import class_prediction, get_response
from preprocess import Tratamento
from preprocess import Correlacao
from random import choice
from models.modelos_intencoes.modelo_suporte.filter import get_solution
from models.modelos_intencoes.modelo_suporte.filter import tf_idf
from models.modelos_intencoes.modelo_suporte.filter import get_response_question
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
            if context == "question_response":
                with open(('logs\\log.json'), 'r', encoding='utf-8') as f:
                    log = json.load(f)
                df_question = pd.read_excel(f'database\\question.xlsx')
                question = ' '.join(log[-1]['response'])
                vetor = 0
                for i in df_question.columns:
                    text_list = [str(j) for j in df_question[i].dropna().tolist() if not isinstance(j, bool)]
                    vetor_encontrado,indice = tf_idf(question,text_list)
                    if vetor_encontrado > vetor:
                        vetor = vetor_encontrado
                        context_question = i
                if context_question in df_question.columns.tolist():
                    input_user = get_response_question(input_user)
                    try:
                        subject = log[-1]["subject"]
                        device = log[-1]["device"]
                        interface = log[-1]["interface"]
                        model = log[-1]["model"]
                        problem = log[-1]["problem"]
                    except TypeError:
                        with open("models\\modelos_intencoes\\anything_else\\intents.json",'r',encoding="UTF-8") as bd:
                            list_intents = json.load(bd)
                        intent_user = [{"intent": "anything_else", "probability": 1.0}]
                        response = get_response(intent_user, list_intents)
                else:
                    with open("models\\modelos_intencoes\\anything_else\\intents.json",'r',encoding="UTF-8") as bd:
                        list_intents = json.load(bd)
                    intent_user = [{"intent": "anything_else", "probability": 1.0}]
                    response = get_response(intent_user, list_intents)
            else:
                try:
                    subject = log[-1]["subject"]
                except:
                    subject = False
                try:
                    device = log[-1]["device"]
                except:
                    device = False
                try:
                    interface = log[-1]["interface"]
                except:
                    interface = False
                try:
                    model = log[-1]["model"]
                except:
                    model = False
                try:
                    problem = log[-1]["problem"]
                except:
                    problem = False
            subject,device,interface,model,problem,response,opcoes = get_solution(input_user,subject,device,interface,model,problem)
            log_chat.log(input_user,context,response,subject,device,interface,model,problem,opcoes)
    response_bot = []
    if type(response) == list:
        for i in response:
            response_bot.append({"text":i})
    else:
        response_bot.append({"text":response})
    return response_bot

while True:
    a = input(": ")
    if a == "cls":
        break
    else:
        b = chatbot_run(a)
    print(b)