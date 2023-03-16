while True:
    try: 
        import os
        import json
        break
    except ModuleNotFoundError:
        continue
from model.log import Log_chat
from model.extract import class_prediction, get_response
from keras.models import load_model

def chatbot_run(input_user):
    id_response = 0
    # extrai o modelo usando o keras
    # model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Chatbot\\model\\model.h5"))
    model_path = "./model/model.h5"
    model = load_model(model_path)
    # abre o banco de dados para carregar as intenções
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Chatbot\\database\\intents.json"))
    with open(db_path,'r',encoding="UTF-8") as banco:
        intents_db = json.load(banco)
    # # chama função de correlção para filtrar palavras inpróprias
    # vetor = ...
    # # verifica se houve correlação com o intent de palavras/frases inpróprias e define a intenção
    # if vetor > 0.3:
    #     intent_user = "censored"
    #     list_response = get_response(intent_user, intents_db)
    #     response = [id_response]
    # else:
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
    intent_user = class_prediction(input_user, model)
    response = get_response(intent_user, intents_db)
            # list_response = get_response(intent_user, intents_db)
            # response = [id_response]
    # cria um log da conversa
    # Log(intent_user,input_user,list_response,response,id_response)
    return response