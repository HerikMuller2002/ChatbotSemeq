from extract import class_prediction, get_response
from models.modelos_intencoes.modelo_suporte.filter import get_solution
from models.modelos_intencoes.modelo_suporte.filter import tf_idf
from models.modelos_intencoes.modelo_suporte.filter import get_response_question
import json
import pandas as pd
import os
# from preprocess import Tratamento

# with open(('logs\\log.json'), 'r', encoding='utf-8') as f:
#     log = json.load(f)
# opcoes = [Tratamento.preprocess_input(i['valor']).split() for i in log[-1]['opcoes']]
# opcoes = [elemento for lista_interna in opcoes for elemento in lista_interna]
# input_user = "camera"
# verificacao_input = [i for i in input_user.split() if i in opcoes]
# if verificacao_input == True or verificacao_input:
#     print(verificacao_input)



def log1(pattern,context,response,subject,device,interface,model,problem,opcoes):
    if os.path.isfile(os.path.join('log.json')):
        with open(os.path.join('log.json'), 'r+', encoding='utf-8') as f:
            log = json.load(f)
            log.append({
                "pattern": pattern,
                "context": context,
                "response": response,
                "subject": subject,
                "device": device,
                "interface": interface,
                "model": model,
                "problem": problem,
                "opcoes": opcoes
            })
            f.seek(0)
            json.dump(log, f, indent=4)
    else:
        with open(os.path.join('log.json'), 'w', encoding='utf-8') as f:
            log = [{
                "pattern": pattern,
                "context": context,
                "response": response,
                "subject": subject,
                "device": device,
                "interface": interface,
                "model": model,
                "problem": problem,
                "opcoes": opcoes
            }]
            json.dump(log, f, indent=4)


def chatbot_run(input_user,context="solution"):
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



    with open(('log.json'), 'r', encoding='utf-8') as f:
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
        df_question = pd.read_excel('database\\respostas_perguntas.xlsx')
        mask = df_question.isin([input_user]).any()
        # obtém o nome da coluna em que a entrada do usuário está presente
        column_name = mask.idxmax()
        if column_name == "pattern_positive":
            ...
        if len(log[-1]['opcoes']) == 1:
            print(log[-1]['opcoes'][0]['valor'])
        else:
            print("varios")

    subject,device,interface,model,problem,response,opcoes = get_solution(input_user,subject,device,interface,model,problem)
    print()
    a = [subject,device,interface,model,problem,response,opcoes]
    for i in a:
        print(i)
    # log1(input_user,context,response,subject,device,interface,model,problem,opcoes)

    response_bot = []
    if type(response) == list:
        for i in response:
            response_bot.append({"text":i})
    else:
        response_bot.append({"text":response})
    return response_bot




while True:
    b = input(": ")
    if b == "cls":
        break
    else:
        a = chatbot_run(b)