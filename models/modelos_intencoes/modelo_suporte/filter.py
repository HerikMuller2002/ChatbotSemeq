import pandas as pd
import numpy as np
import os
import sys
import re
from json import load
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

def tf_idf(user_input, dados):
    if isinstance(dados, pd.DataFrame):
        dataframe = dados.astype(str)
        list_text_db = dataframe.to_numpy().flatten().tolist()
    elif isinstance(dados, list):
        list_text_db = dados.copy()
    else:
        raise ValueError("Os dados precisam ser uma lista ou um DataFrame.")
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
    df_subject = df.loc[df['subject'] == subject].drop('subject', axis=1)
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
    df_device = df_subject.loc[df_subject['device'] == device].drop(['subject','device'], axis=1)
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
    df_interface = df_device.loc[df_device['interface'] == interface].drop(['subject','device','interface'], axis=1)
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
    df_model = df_interface.loc[df_interface['model'] == model].drop(['subject','device','interface','model'], axis=1)
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
    df_question = pd.read_excel(f'{pai_path}\\database\\question.xlsx')
    if level_dict:
        greater_than = [elemento for elemento in level_dict if elemento['vetor'] > 0.7]
        if level_dict[0]['vetor'] < 0.7 or len(greater_than) > 1:
            if len(greater_than) > 1:
                if len(greater_than) == 2:
                    question = sub("[_]", f"{greater_than[0]['level']} ou {greater_than[1]['level']}", df_question.loc[0,'question_doubt'])
                elif len(greater_than) == 3:
                    question = sub("[_]", f"{greater_than[0]['level']}, {greater_than[1]['level']} ou {greater_than[2]['level']}", df_question.loc[0,'question_doubt'])
                question = question.split("?")
                question = [x + "?" for x in question[:-1]] + [question[-1]]
                question.pop()
                dict_list_option = []
                count = 0
                for i in greater_than:
                    count += 1
                    dict_list_option.append({"opcao":str(count),"valor":i['level']})
            elif level_dict[0]['vetor'] > 0.1:
                question = sub("[_]", f"{level_dict[0]['level']}", df_question.loc[0,'question_uncertainty'])
                question = question.split("?")
                question = [x + "?" for x in question[:-1]] + [question[-1]]
                question.pop()
                dict_list_option = [{"opcao":'1',"valor":level_dict[0]['level']}]
            else:
                if len(list_level) == 1:
                    question = sub("[_]", f"{list_level[0]}", df_question.loc[0,'question_uncertainty'])
                    question = question.split("?")
                    question = [x + "?" for x in question[:-1]] + [question[-1]]
                    question.pop()
                    dict_list_option = [{"opcao":'1',"valor":list_level[0]}]
                else:
                    count = 0
                    dict_list_option = []
                    list_option = []
                    for i in list_level:
                        count += 1
                        text = f'{count}-{i.capitalize()}'
                        dict_list_option.append({"opcao":str(count),"valor":i})
                        list_option.append(text)
                    options = "¬".join(list_option)
                    question = df_question.loc[0,'question_options']
                    question = question.split("_")
                    question.append(options)
                    question[-2],question[-1] = question[-1], question[-2]
        else:
            question = False
            dict_list_option = False
    else:
        count = 0
        dict_list_option = []
        list_option = []
        for i in list_level:
            count += 1
            text = f'{count}-{i.capitalize()}'
            dict_list_option.append({"opcao":str(count),"valor":i})
            list_option.append(text)
        options = "¬".join(list_option)
        question = df_question.loc[0,'question_options']
        question = question.split("_")
        question.append(options)
        question[-2],question[-1] = question[-1], question[-2]
    return question,dict_list_option

###################################################

# carrega o dataframe
df_response = pd.read_excel(f'{pai_path}\\database\\respostas_perguntas.xlsx')

# função para verificar em qual coluna o input tem maior similaridade
def get_column(input_text):
    df1 = df_response[['pattern_positive']]
    df2 = df_response[['pattern_negative']]
    similarity1,indice = tf_idf(input_text, df1)
    similarity2,indice = tf_idf(input_text, df2)
    if similarity1 > similarity2:
        max_column = ' '.join(df1.columns)
    elif similarity1 < similarity2:
        max_column = ' '.join(df2.columns)
    else:
        max_column = False
    return max_column

def get_response_question(input_user):
    # verificando qual era a pergunta
    if os.path.isfile(os.path.join(pai_path,'logs\\log.json')):
        with open(('logs\\log.json'), 'r', encoding='utf-8') as log_chat:
            log = load(log_chat)
        df_question = pd.read_excel(f'{pai_path}\\database\\question.xlsx')
        question = ' '.join(log[-1]['response'])
        vetor = 0
        for i in df_question.columns:
            text_list = [str(j) for j in df_question[i].dropna().tolist() if not isinstance(j, bool)]
            vetor_encontrado,indice = tf_idf(question,text_list)
            if vetor_encontrado > vetor:
                vetor = vetor_encontrado
                context_question = i
        # verificando se a resposta é sim ou não
        column = get_column(input_user)
        if context_question == 'question_uncertainty':
            if column == 'pattern_positive':
                response_user = log[-1]['opcoes'][0]['valor']
            else:
                response_user = False
        elif context_question == 'question_doubt':
            for i in log[-1]['opcoes']:
                if i['opcao'] == input_user or i['valor'] == input_user:
                    response_user = i['valor']
                else:
                    response_user = False
        else:
            for i in log[-1]['opcoes']:
                if i['opcao'] == input_user or i['valor'] == input_user:
                    response_user = i['valor']
                else:
                    response_user = False
    else:
        response_user = False
    try:
        return response_user
    except UnboundLocalError:
        return False

def get_solution(input_user,subject,device,interface,model,problem):
    if not input_user:
        with open(('logs\\log.json'), 'r', encoding='utf-8') as log_chat:
            log = load(log_chat)
        original_dict = log[-1]
        new_dict = {
            "subject": original_dict["subject"],
            "device": original_dict["device"],
            "interface": original_dict["interface"],
            "model": original_dict["model"],
            "problem": original_dict["problem"]
            }
        subcontext = None
        for chave, valor in reversed(list(new_dict.items())):
            if type(valor) == str:
                subcontext = chave
                value_subcontext = valor
                break
        if subcontext == 'subject':
            troubleshooting = df.groupby('subject')
            list_subject = [i for i,j in troubleshooting]
            question,dict_list_option = get_question(False,list_subject)

            response = question
            return subject,device,interface,model,problem,response,dict_list_option
        
        elif subcontext == 'device':
            df_subject = df.loc[df['subject'] == subject]
            df_subject = df_subject.drop('subject', axis=1)
            troubleshooting = df_subject.groupby('device')
            list_device = [i for i,j in troubleshooting]
            question,dict_list_option = get_question(False,list_device)

            response = question
            return subject,device,interface,model,problem,response,dict_list_option
        
        elif subcontext == 'interface':
            df_subject = df.loc[df['subject'] == subject]
            df_device = df_subject.loc[df_subject['device'] == device]
            df_device = df_subject.drop(['subject','device'], axis=1)
            troubleshooting = df_device.groupby('interface')
            list_interface = [i for i,j in troubleshooting]
            question,dict_list_option = get_question(False,list_interface)

            response = question
            return subject,device,interface,model,problem,response,dict_list_option
        
        elif subcontext == 'model':
            df_subject = df.loc[df['subject'] == subject]
            df_device = df_subject.loc[df_subject['device'] == device]
            df_interface = df_device.loc[df_device['interface'] == interface]
            df_interface = df_interface.drop(['subject','device','interface'], axis=1)
            troubleshooting = df_interface.groupby('model')
            list_model = [i for i,j in troubleshooting]
            question,dict_list_option = get_question(False,list_model)

            response = question
            return subject,device,interface,model,problem,response,dict_list_option
        
        elif subcontext == 'problem':
            df_subject = df.loc[df['subject'] == subject]
            df_device = df_subject.loc[df_subject['device'] == device]
            df_interface = df_device.loc[df_device['interface'] == interface]
            df_model = df_interface.loc[df_interface['model'] == model]
            df_model = df_model.drop(['subject','device','interface','model'], axis=1)
            troubleshooting = df_model.groupby('problem')
            list_problem = [i for i,j in troubleshooting]
            question,dict_list_option = get_question(False,list_problem)

            response = question
            return subject,device,interface,model,problem,response,dict_list_option
    
    else:
        if not subject:
            # assunto
            subject_dict,list_subject = match_subject(input_user)
            question,dict_list_option = get_question(subject_dict,list_subject)
            subject = subject_dict[0]['level']
            if question:
                response = question
                return subject,device,interface,model,problem,response,dict_list_option
            else:
                device = False
        else:
            subject_dict = [{"vetor":1,"level":subject}]
        if not device:
            # device
            device_dict,list_device = match_device(input_user,subject_dict[0]['level'])
            question,dict_list_option = get_question(device_dict,list_device)
            device = device_dict[0]['level']
            if question:
                response = question
                return subject,device,interface,model,problem,response,dict_list_option
            else:
                interface = False
        else:
            device_dict = [{"vetor":1,"level":device}]
        if not interface:
            # interface
            interface_dict,list_interface = match_interface(input_user,subject_dict[0]['level'],device_dict[0]['level'])
            question,dict_list_option = get_question(interface_dict,list_interface)
            interface = interface_dict[0]['level']
            if question:
                response = question
                return subject,device,interface,model,problem,response,dict_list_option
            else:
                model = False
        else:
            interface_dict = [{"vetor":1,"level":interface}]
        if not model:        
            # model
            model_dict,list_model = match_model(input_user,subject_dict[0]['level'],device_dict[0]['level'],interface_dict[0]['level'])
            question,dict_list_option = get_question(model_dict,list_model)
            model = model_dict[0]['level']
            if question:
                response = question
                return subject,device,interface,model,problem,response,dict_list_option
            else:
                problem = False
        else:
            model_dict = [{"vetor":1,"level":model}]
        # if not problem:
        # problem
        problem_dict,list_problem = match_problem(input_user,subject_dict[0]['level'],device_dict[0]['level'],interface_dict[0]['level'],model_dict[0]['level'])
        question,dict_list_option = get_question(problem_dict,list_problem)
        problem = problem_dict[0]['level']
        if question:
            response = question
            return subject,device,interface,model,problem,response,dict_list_option
        else:
            response = []
            list_response = ["Entendi, você está com o seguinte problema: _","Aqui está uma possível solução...","Caso o problema não seja resolvido, por favor, abra um chamado para o Service Desk da Semeq pelo e-mail servicedesk@semeq.com. Espero ter ajudado!"]
            for i in list_response:
                i = sub("[_]", f"{problem}", i)
                response.append(i)
                if "..." in i:
                    df_solution = pd.read_excel('database\\troubleshooting.xlsx')
                    solution = df_solution.loc[(df_solution['subject'] == subject) &
                            (df_solution['device'] == device) &
                            (df_solution['interface'] == interface) &
                            (df_solution['model'] == model) &
                            (df_solution['problem'] == problem),
                            'solution'].iloc[0]
                    response.append(solution)

            return subject,device,interface,model,problem,response,dict_list_option