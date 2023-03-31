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

        elif context == "unsolved":
            with open("models\\modelos_intencoes\\unsolved\\intents.json",'r',encoding="UTF-8") as bd:
                list_intents = json.load(bd)
            intent_user = [{"intent": "unsolved", "probability": 1.0}]
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
            with open(('logs\\log.json'), 'r', encoding='utf-8') as f:
                log = json.load(f)
            try:
                verificacao = log[-1]['context']
                if verificacao == "question_response" or verificacao == "solution":
                    verificacao_input = True
                else:
                    verificacao_input = False
            except IndexError:
                verificacao_input = False

            if context == "question_response" and verificacao_input or context == "solution" and verificacao_input:
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
                    with open(('logs\\log.json'), 'r', encoding='utf-8') as log_file:
                        log = json.load(log_file)
                    ultimo_dicionario = log[-1]
                    if not get_response_question(input_user):
                        # Percorrer o último dicionário de trás para frente
                        for chave, valor in reversed(list(ultimo_dicionario.items())):
                            if isinstance(valor, str):
                                # Armazenar a última chave que tem valor string
                                subcontext = chave
                                value_subcontext = valor
                                # Substituir o valor string por False
                                ultimo_dicionario[chave] = False
                                break
                        input_user = False
                    else:
                        input_user = get_response_question(input_user)
                    #     # Criar um novo dicionário com as chaves e valores necessários
                    #     new_dict = {
                    #         "subject": ultimo_dicionario["subject"],
                    #         "device": ultimo_dicionario["device"],
                    #         "interface": ultimo_dicionario["interface"],
                    #         "model": ultimo_dicionario["model"],
                    #         "problem": ultimo_dicionario["problem"]
                    #     }
                    #     # Percorrer o novo dicionário em busca de valores False
                    #     for chave, valor in new_dict.items():
                    #         if valor == False:
                    #             # Substituir o valor False no último dicionário
                    #             ultimo_dicionario[chave] = get_response_question(input_user)
                    #     input_user = ultimo_dicionario['pattern']
                    # # Salvar o último dicionário modificado de volta no arquivo JSON
                    # with open(('logs\\log.json'), 'w', encoding='utf-8') as log_file:
                    #     json.dump(list(log), log_file)

                    # with open(('logs\\log.json'), 'r', encoding='utf-8') as log_file:
                    #     log = json.load(log_file)
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
                        response_bot = []
                        if type(response) == list:
                            for i in response:
                                response_bot.append({"text":i})
                        else:
                            response_bot.append({"text":response})
                        return response_bot
                else:
                    with open("models\\modelos_intencoes\\anything_else\\intents.json",'r',encoding="UTF-8") as bd:
                        list_intents = json.load(bd)
                    intent_user = [{"intent": "anything_else", "probability": 1.0}]
                    response = get_response(intent_user, list_intents)
                    response_bot = []
                    if type(response) == list:
                        for i in response:
                            response_bot.append({"text":i})
                    else:
                        response_bot.append({"text":response})
                    return response_bot
            elif context == "question_response" and not verificacao_input:
                with open("models\\modelos_intencoes\\anything_else\\intents.json",'r',encoding="UTF-8") as bd:
                    list_intents = json.load(bd)
                intent_user = [{"intent": "anything_else", "probability": 1.0}]
                response = get_response(intent_user, list_intents)
                response_bot = []
                if type(response) == list:
                    for i in response:
                        response_bot.append({"text":i})
                else:
                    response_bot.append({"text":response})
                return response_bot
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