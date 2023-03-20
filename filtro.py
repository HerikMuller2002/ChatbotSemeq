import json
import os
from extract import class_prediction, get_response

def analysis(input_user):
    with open("models\\modelos_intencoes\\modelo_solution\\intents.json",'r',encoding="UTF-8") as bd:
        list_intents = json.load(bd)
    model_path = "models\\modelos_intencoes\\modelo_solution\\model.h5"
    words_path = "models\\modelos_intencoes\\modelo_solution\\words.pkl"
    classes_path = "models\\modelos_intencoes\\modelo_solution\\classes.pkl"
    subject = class_prediction(input_user, model_path,words_path,classes_path)
    if subject[0]['probability'] > 0.7:
        # chama o póximo nível
        with open(f"models\\modelos_intencoes\\modelo_solution\\intents.json",'r',encoding="UTF-8") as bd:
            list_intents = json.load(bd)
        model_path = "models\\modelos_intencoes\\modelo_solution\\model.h5"
        words_path = "models\\modelos_intencoes\\modelo_solution\\words.pkl"
        classes_path = "models\\modelos_intencoes\\modelo_solution\\classes.pkl"
    else:
        # perguntas
        ...