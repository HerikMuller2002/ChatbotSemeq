import json
import os
from extract import class_prediction, get_response

def analysis(input_user):
    # assunto
    with open("models\\modelos_intencoes\\modelo_solution\\intents.json",'r',encoding="UTF-8") as bd:
        list_subject = json.load(bd)
    model_path = "models\\modelos_intencoes\\modelo_solution\\model.h5"
    words_path = "models\\modelos_intencoes\\modelo_solution\\words.pkl"
    classes_path = "models\\modelos_intencoes\\modelo_solution\\classes.pkl"
    intent_subject = class_prediction(input_user, model_path,words_path,classes_path)
    subject = get_response(intent_subject, list_subject)
    if subject == "anything_else":
        with open("models\\modelos_intencoes\\anything_else\\intents.json",'r',encoding="UTF-8") as bd:
            list_intents = json.load(bd)
        intent_user = [{"intent": "anything_else", "probability": 1.0}]
        response = get_response(intent_user, list_intents)
    else:
        # device
        with open(f"models\\modelos_intencoes\\modelo_solution\\modelo_device\\{subject}\\intents.json",'r',encoding="UTF-8") as bd:
            list_device = json.load(bd)
        model_path = f"models\\modelos_intencoes\\modelo_solution\\modelo_device\\{subject}\\model.h5"
        words_path = f"models\\modelos_intencoes\\modelo_solution\\modelo_device\\{subject}\\words.pkl"
        classes_path = f"models\\modelos_intencoes\\modelo_solution\\modelo_device\\{subject}\\classes.pkl"
        intent_device = class_prediction(input_user, model_path,words_path,classes_path)
        device = get_response(intent_device, list_device)
        if device == "anything_else":
            with open("models\\modelos_intencoes\\anything_else\\intents.json",'r',encoding="UTF-8") as bd:
                list_intents = json.load(bd)
            intent_user = [{"intent": "anything_else", "probability": 1.0}]
            response = get_response(intent_user, list_intents)
        else:
            # interface
            with open(f"models\\modelos_intencoes\\modelo_solution\\modelo_device\\{subject}\\{device}\\intents.json",'r',encoding="UTF-8") as bd:
                list_device = json.load(bd)
            model_path = f"models\\modelos_intencoes\\modelo_solution\\modelo_device\\{subject}\\{device}\\model.h5"
            words_path = f"models\\modelos_intencoes\\modelo_solution\\modelo_device\\{subject}\\{device}\\words.pkl"
            classes_path = f"models\\modelos_intencoes\\modelo_solution\\modelo_device\\{subject}\\{device}\\classes.pkl"
            intent_device = class_prediction(input_user, model_path,words_path,classes_path)
            interface = get_response(intent_device, list_device)
            if interface == "anything_else":
                with open("models\\modelos_intencoes\\anything_else\\intents.json",'r',encoding="UTF-8") as bd:
                    list_intents = json.load(bd)
                intent_user = [{"intent": "anything_else", "probability": 1.0}]
                response = get_response(intent_user, list_intents)
            else:
                response = interface

    return subject + " / " + device + " / " + response