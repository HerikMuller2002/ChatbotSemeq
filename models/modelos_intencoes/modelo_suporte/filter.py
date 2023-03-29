import pandas as pd
import numpy as np
import os
import sys
from regex import sub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0,pai_path)

df = pd.read_excel(f'{pai_path}\\database\\troubleshooting.xlsx')

def preprocess_input(text):
    text = text.lower().strip()
    # tirar pontuações, acentos e espaços extras
    text = sub('[áàãâä]', 'a', sub('[éèêë]', 'e', sub('[íìîï]', 'i', sub('[óòõôö]', 'o', sub('[úùûü]', 'u', text)))))
    # tirar espaços em branco
    text = sub(r'\s+', ' ',text)
    return text

def preprocess_list(list_text):
    new_list = []
    for text in list_text:
        text = preprocess_input(text)
        if ',' in text or '\\' in text or '/' in text:
            new_texts = text.split(',') + text.split('\\') + text.split('/')
            for new_text in new_texts:
                new_list.append(new_text)
        else:
            new_texts = text.split()
            for new_text in new_texts:
                new_list.append(new_text)
            new_list.append(text)
    return new_list

def tf_idf(user_input, dataframe):
    dataframe = dataframe.astype(str)
    list_text_db = dataframe.to_numpy().flatten().tolist()
    list_text_db = preprocess_list(list_text_db)
    list_text_db.append(user_input)
    tfidf = TfidfVectorizer()
    palavras_vetorizadas = tfidf.fit_transform(list_text_db)
    similaridade = cosine_similarity(palavras_vetorizadas[-1], palavras_vetorizadas)
    vetor_similar = similaridade.flatten()
    vetor_similar.sort()
    vetor_similar = np.delete(vetor_similar, -1)
    if vetor_similar.size == 0:
        return 0, False
    vetor_encontrado = vetor_similar[-1]
    indice_sentenca = np.where(similaridade == vetor_encontrado)[0][-1]
    return vetor_encontrado, indice_sentenca

###################################################

def match_subject(input_user):
    troubleshooting = df.groupby('subject')
    list_subject = [i for i,j in troubleshooting]
    vetores = []
    for i in list_subject:
        list_subject_df = df.loc[df['subject']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_subject_df)
        vetores.append(vetor_encontrado)
    # Criar lista de dicionários
    dict_list = []
    # Percorrer vetores e índices correspondentes
    for i, vetor in enumerate(vetores):
            # Encontrar subject correspondente
        subject = list_subject[i]
        # Adicionar dicionário na lista
        dict_list.append({"vetor":vetor,"level":subject})
    if len(dict_list) == 0:
        dict_list = [{"vetor":0,"level":False}]
    else:
        dict_list = sorted(dict_list, key=lambda x: x["vetor"], reverse=True)
        while len(dict_list) > 3:
            dict_list.pop()
    return dict_list,list_subject

def match_device(input_user,subject):
    df_subject = df.loc[df['subject'] == subject]
    df_subject = df_subject.drop('subject', axis=1)
    troubleshooting = df_subject.groupby('device')
    list_device = [i for i,j in troubleshooting]
    vetores = []
    for i in list_device:
        list_device_df = df_subject.loc[df_subject['device']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_device_df)
        vetores.append(vetor_encontrado)
    # Criar lista de dicionários
    dict_list = []
    # Percorrer vetores e índices correspondentes
    for i, vetor in enumerate(vetores):
        if vetor > 0.7:
            # Encontrar subject correspondente
            device = list_device[i]
            
            # Adicionar dicionário na lista
            dict_list.append({"vetor":vetor,"level":device})
    if len(dict_list) == 0:
        dict_list = [{"vetor":0,"level":False}]
    else:
        dict_list = sorted(dict_list, key=lambda x: x["vetor"], reverse=True)
        if len(dict_list) > 3:
            dict_list.pop()
    return dict_list,list_device

def match_interface(input_user,subject,device):
    df_subject = df.loc[df['subject'] == subject]
    df_device = df_subject.loc[df_subject['device'] == device]
    df_device = df_subject.drop(['subject','device'], axis=1)
    troubleshooting = df_device.groupby('interface')
    list_interface = [i for i,j in troubleshooting]
    vetores = []
    for i in list_interface:
        list_interface_df = df_device.loc[df_device['interface']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_interface_df)
        vetores.append(vetor_encontrado)
    # Criar lista de dicionários
    dict_list = []
    # Percorrer vetores e índices correspondentes
    for i, vetor in enumerate(vetores):
        if vetor > 0.7:
            # Encontrar subject correspondente
            device = list_interface[i]
            
            # Adicionar dicionário na lista
            dict_list.append({"vetor":vetor,"level":device})
    if len(dict_list) == 0:
        dict_list = [{"vetor":0,"level":False}]
    else:
        dict_list = sorted(dict_list, key=lambda x: x["vetor"], reverse=True)
        if len(dict_list) > 3:
            dict_list.pop()
    return dict_list,list_interface

def match_model(input_user,subject,device,interface):
    df_subject = df.loc[df['subject'] == subject]
    df_device = df_subject.loc[df_subject['device'] == device]
    df_interface = df_device.loc[df_device['interface'] == interface]
    df_interface = df_interface.drop(['subject','device','interface'], axis=1)
    troubleshooting = df_interface.groupby('model')
    list_model = [i for i,j in troubleshooting]
    vetores = []
    for i in list_model:
        list_model_df = df_interface.loc[df_interface['model']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_model_df)
        vetores.append(vetor_encontrado)
    # Criar lista de dicionários
    dict_list = []
    # Percorrer vetores e índices correspondentes
    for i, vetor in enumerate(vetores):
        if vetor > 0.7:
            # Encontrar subject correspondente
            model = list_model[i]
            
            # Adicionar dicionário na lista
            dict_list.append({"vetor":vetor,"level":model})
    if len(dict_list) == 0:
        dict_list = [{"vetor":0,"level":False}]
    else:
        dict_list = sorted(dict_list, key=lambda x: x["vetor"], reverse=True)
        if len(dict_list) > 3:
            dict_list.pop()
    return dict_list,list_model

def match_problem(input_user,subject,device,interface,model):
    df_subject = df.loc[df['subject'] == subject]
    df_device = df_subject.loc[df_subject['device'] == device]
    df_interface = df_device.loc[df_device['interface'] == interface]
    df_model = df_interface.loc[df_interface['model'] == model]
    df_model = df_model.drop(['subject','device','interface','model'], axis=1)
    troubleshooting = df_model.groupby('problem')
    list_problem = [i for i,j in troubleshooting]
    vetores = []
    for i in list_problem:
        list_problem_df = df_model.loc[df_model['problem']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_problem_df)
        vetores.append(vetor_encontrado)
    # Criar lista de dicionários
    dict_list = []
    # Percorrer vetores e índices correspondentes
    for i, vetor in enumerate(vetores):
        if vetor > 0.7:
            # Encontrar subject correspondente
            problem = list_problem[i]
            
            # Adicionar dicionário na lista
            dict_list.append({"vetor":vetor,"level":problem})
    if len(dict_list) == 0:
        dict_list = [{"vetor":0,"level":False}]
    else:
        dict_list = sorted(dict_list, key=lambda x: x["vetor"], reverse=True)
        if len(dict_list) > 3:
            dict_list.pop()
    return dict_list,list_problem

###################################################

def get_question(level_dict,list_level):
    greater_than = [elemento for elemento in level_dict if elemento['vetor'] > 0.7]
    if level_dict[0]['vetor'] < 0.7 or len(greater_than) > 1:
        df_question = pd.read_excel(f'{pai_path}\\database\\question.xlsx')
        if len(greater_than) > 1:
            if len(greater_than) == 2:
                question = sub("[_]", f"{greater_than[0]['level']} ou {greater_than[1]['level']}", df_question.loc[0,'question_doubt'])
            elif len(greater_than) == 3:
                question = sub("[_]", f"{greater_than[0]['level']}, {greater_than[1]['level']} ou {greater_than[2]['level']}", df_question.loc[0,'question_doubt'])
            question = question.split("?")
            question = [x + "?" for x in question[:-1]] + [question[-1]]
            question.pop()
        elif level_dict[0]['vetor'] > 0.1:
            question = sub("[_]", f"{level_dict[0]['level']}", df_question.loc[0,'question_uncertainty'])
            question = question.split("?")
            question = [x + "?" for x in question[:-1]] + [question[-1]]
            question.pop()
        else:
            if len(list_level) < 2:
                question = sub("[_]", f"{list_level[0]}", df_question.loc[0,'question_uncertainty'])
                question = question.split("?")
                question = [x + "?" for x in question[:-1]] + [question[-1]]
                question.pop()
            else:
                count = 0
                dict_list_option = []
                list_option = []
                for i in list_level:
                    count += 1
                    text = f'{count}-{i.capitalize()}'
                    dict_list_option.append({"opcao":count,"valor":i})
                    list_option.append(text)
                options = ", ".join(list_option)
                question = df_question.loc[0,'question_options']
                question = question.split("_")
                question.append(options)
                question[-2],question[-1] = question[-1], question[-2]
    else:
        question = False
    return question

def get_solution(input_user):
    subject = False
    device = False
    interface = False
    model = False
    problem = False

    # assunto
    subject_dict,list_subject = match_subject(input_user)
    question = get_question(subject_dict,list_subject)
    subject = subject_dict[0]['level']
    if question:
        response = question
        return subject,device,interface,model,problem,response
    else:
        # device
        device_dict,list_device = match_device(input_user,subject_dict[0]['level'])
        question = get_question(device_dict,list_device)
        device = device_dict[0]['level']
        if question:
            response = question
            return subject,device,interface,model,problem,response
        else:
            # interface
            interface_dict,list_interface = match_interface(input_user,subject_dict[0]['level'],device_dict[0]['level'])
            question = get_question(interface_dict,list_interface)
            interface = interface_dict[0]['level']
            if question:
                response = question
                return subject,device,interface,model,problem,response
            else:
                # model
                model_dict,list_model = match_model(input_user,subject_dict[0]['level'],device_dict[0]['level'],interface_dict[0]['level'])
                question = get_question(model_dict,list_model)
                model = model_dict[0]['level']
                if question:
                    response = question
                    return subject,device,interface,model,problem,response
                else:
                    # problem
                    problem_dict,list_problem = match_problem(input_user,subject_dict[0]['level'],device_dict[0]['level'],interface_dict[0]['level'],model_dict[0]['level'])
                    question = get_question(problem_dict,list_problem)
                    problem = problem_dict[0]['level']
                    if question:
                        response = question
                        return subject,device,interface,model,problem,response
                    else:
                        response = problem
                        return subject,device,interface,model,problem,response