import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import sys
pai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0,pai_path)

def tf_idf(user_input, dataframe):
    dataframe = dataframe.astype(str)
    list_text_db = dataframe.to_numpy().flatten().tolist()
    list_text_db.append(user_input)
    tfidf = TfidfVectorizer()
    list_text_db = list(filter(None, list_text_db))
    palavras_vetorizadas = tfidf.fit_transform(list_text_db)
    similaridade = cosine_similarity(palavras_vetorizadas[-1], palavras_vetorizadas)
    vetor_similar = similaridade.flatten()
    vetor_similar.sort()
    vetor_similar = np.delete(vetor_similar, -1)
    if vetor_similar.size == 0:
        return 0, 0
    vetor_encontrado = vetor_similar[-1]
    indice_sentenca = np.where(similaridade == vetor_encontrado)[0][-1]
    return vetor_encontrado, indice_sentenca

def get_response_question(log_conversation,input_user):
    original_dict = log_conversation[-1]
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
    df = pd.read_excel(f'{pai_path}\\database\\respostas_perguntas.xlsx')
    # cria um dicionário para armazenar os valores de similaridade de cada coluna
    similarity_dict = {}
    # percorre as colunas do dataframe
    for column in df.columns:
        # cria um vetor com os textos da coluna e do subcontext
        text_list = [str(i) for i in df[column].dropna().tolist() if not isinstance(i, bool)] + [subcontext]
        # instância o vetorizador tf-idf
        vectorizer = TfidfVectorizer()
        # calcula a matriz de frequência com o vetorizador
        tf_idf_matrix = vectorizer.fit_transform(text_list)
        # calcula a similaridade do input do usuário com a coluna atual
        input_vector = vectorizer.transform([input_user])
        similarity = (input_vector * tf_idf_matrix.T).A[0].max()
        # armazena a similaridade no dicionário
        similarity_dict[column] = similarity
    # obtém a coluna com maior similaridade
    max_column = max(similarity_dict, key=similarity_dict.get)
    if max_column == 'pattern_positive':
        # percorre a lista da direita para a esquerda usando reversed() e retorna o primeiro elemento que não é booleano
        level_intent = []
        for chave,valor in new_dict.items():
            if type(valor) != bool:
                level_intent.append(valor)
        new_response = ' '.join(level_intent)

    return max_column, subcontext, new_response


# json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
# if os.path.exists(json_path):
#     if os.path.isfile(os.path.join(json_path, 'log.json')):
#         with open(os.path.join(json_path, 'log.json'), 'r', encoding='utf-8') as f:
#             log = json.load(f)
# max_column, subcontext, new_response = get_response_question(log,'sim')
# print(new_response)

# def get_pre_response(log_conversation,input_user,content):
#     json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
#     if os.path.isfile(os.path.join(json_path, 'log.json')):
#         df = pd.read_excel('\\Users\\Semeq\\Desktop\\ChatbotSemeq\\database\\respostas_perguntas.xlsx')
#         df_positive = df[['pattern_positive']]
#         df_negative = df[['pattern_negative']]
#         vetor_positive, indice_sentenca = tf_idf(input_user, df_positive)
#         vetor_negative, indice_sentenca = tf_idf(input_user, df_negative)
#         ultimo_nao_boolean = None
#         for chave,valor in new_dict.items():
#             if type(valor) != bool:
#                 ultimo_nao_boolean = chave
#         if vetor_positive > vetor_negative:
#             original_dict = log_conversation[-1]
#             last_input = original_dict["pattern"]
#             new_dict = {
#             "subject": original_dict["subject"],
#             "device": original_dict["device"],
#             "interface": original_dict["interface"],
#             "model": original_dict["model"],
#             "problem": original_dict["problem"]
#             }
#             # percorre a lista da direita para a esquerda usando reversed() e retorna o primeiro elemento que não é booleano
#             level_intent = []
#             for chave,valor in new_dict.items():
#                 if type(valor) != bool:
#                     level_intent.append(valor)
#             levels_intents = ' '.join(level_intent)
#             return levels_intents + ' ' + content + ' ', {ultimo_nao_boolean:True}
#         elif vetor_negative > vetor_positive:
#             original_dict = log_conversation[-1]
#             last_input = original_dict["pattern"]
#             new_dict = {
#             "subject": original_dict["subject"],
#             "device": original_dict["device"],
#             "interface": original_dict["interface"],
#             "model": original_dict["model"],
#             "problem": original_dict["problem"]
#             }
#             # percorre a lista da direita para a esquerda usando reversed() e retorna o primeiro elemento que não é booleano
#             level_intent = []
#             for chave,valor in new_dict.items():
#                 if type(valor) != bool:
#                     level_intent.append(valor)
#             levels_intents = ' '.join(level_intent)
#             return levels_intents + ' ', {ultimo_nao_boolean:False}
#     else:
#         return input_user,{"verify_passar":False}

def get_subject(input_user, first_question=True,user_question_response=False,subcontext=False,value_subcontext=False):
    # assunto
    df = pd.read_excel(f'{pai_path}\\database\\troubleshooting.xlsx')
    troubleshooting = df.groupby('subject')
    list_subject = [i for i,j in troubleshooting]
    vetor = 0
    linha = None
    for i in list_subject:
        list_subject_df = df.loc[df['subject']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_subject_df)
        if vetor_encontrado > vetor:
            vetor = vetor_encontrado
            linha = list_subject_df.iloc[indice_sentenca]
    subject = linha['subject']
    # if vetor < 0.7:
    #     # perguntas
    #     df_questions = pd.read_excel(f'{pai_path}\\database\\question.xlsx')
    #     list_response = []
    #     if first_question:
    #         first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
    #         list_response.append(first_question)
    #         first_question = False
    #         if vetor > 0.3:
    #             df_moment1 = df_questions.loc[df_questions['subject'] == subject]
    #             linha = df_moment1.iloc[indice_sentenca]
    #             list_response.append(linha['subject_question'])
    #             response = list_response
    #             subject = False
    #             device = False
    #             interface = False
    #             model = False
    #             problem = False
    #             return response, subject, device, interface, model, problem,first_question
    #         if vetor < 0.3:
    #             intro = "Qual dessas opções o seu problema melhor se enquadra?"
    #             list_response.append(intro)
    #             count = 0
    #             for i in list_subject:
    #                 count += 1
    #                 i_subject = f'{count} - {i.capitalize()}'
    #                 list_response.append(i_subject)
    #             response = list_response
    #             subject = False
    #             device = False
    #             interface = False
    #             model = False
    #             problem = False
    #             return response, subject, device, interface, model, problem, first_question
    # else:
    response, device, interface, model, problem,first_question = get_device(input_user,df,subject,first_question)
    response = f'assunto: {subject} , device: {device} , interface: {interface} , modelo: {model} , problema: {problem}'
    return response, subject, device, interface, model, problem, first_question

def get_device(input_user,df,subject,first_question):
    # device
    df_subject = df.loc[df['subject'] == subject]
    troubleshooting2 = df_subject.groupby('device')
    list_device = [i for i,j in troubleshooting2]
    vetor = 0
    linha = None
    for i in list_device:
        list_device_df = df_subject.loc[df_subject['device']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_device_df)
        if vetor_encontrado > vetor:
            vetor = vetor_encontrado
            linha = list_device_df.iloc[indice_sentenca]
    device = linha['device']
#     if vetor < 0.7:
#         # perguntas
#         df_questions = pd.read_excel(f'{pai_path}\\database\\question.xlsx')
#         list_response = []
#         if first_question:
#             first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
#             list_response.append(first_question)
#             first_question = False
#         if vetor > 0.3:
#             df_moment1 = df_questions.loc[df_questions['subject'] == subject]
#             df_moment2 = df_moment1.loc[df_moment1['device'] == device]
#             linha =  df_moment2.iloc[indice_sentenca]
#             list_response.append(linha['device_question'])
#             response = list_response
#             device = False
#             interface = False
#             model = False
#             problem = False
#             return response, device, interface, model, problem,first_question
#         else:
#             intro = "Qual dessas opções o seu problema melhor se enquadra?"
#             list_response.append(intro)
#             count = 0
#             for i in list_device:
#                 count += 1
#                 i_device = f'{count} - {i.capitalize()}'
#                 list_response.append(i_device)
#             response = list_response
#             device = False
#             interface = False
#             model = False
#             problem = False
#             return response, device, interface, model, problem,first_question
#     else:
    response, interface, model, problem,first_question = get_interface(input_user,subject,df_subject,device,first_question)
    return response, device, interface, model, problem,first_question

def get_interface(input_user,subject,df_subject,device,first_question):
    # interface
    df_device = df_subject.loc[df_subject['device'] == device]
    troubleshooting3 = df_device.groupby('interface')
    list_interface = [i for i,j in troubleshooting3]
    vetor = 0
    linha = None
    for i in list_interface:
        list_interface_df = df_device.loc[df_device['interface']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_interface_df)
        if vetor_encontrado > vetor:
            vetor = vetor_encontrado
            linha = list_interface_df.iloc[indice_sentenca]
    interface = linha['interface']
#     if vetor < 0.7:
#         # perguntas
#         df_questions = pd.read_excel(f'{pai_path}\\database\\question.xlsx')
#         list_response = []
#         if first_question:
#             first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
#             list_response.append(first_question)
#             first_question = False
#         if vetor > 0.3:
#             df_moment1 = df_questions.loc[df_questions['subject'] == subject]
#             df_moment2 = df_moment1.loc[df_moment1['device'] == device]
#             df_moment3 = df_moment2.loc[df_moment2['interface'] == interface]
#             linha = df_moment3.iloc[indice_sentenca]
#             list_response.append(linha['interface_question'])
#             response = list_response
#             interface = False
#             model = False
#             problem = False
#             return response, interface, model, problem,first_question
#         else:
#             intro = "Qual dessas opções o seu problema melhor se enquadra?"
#             list_response.append(intro)
#             count = 0
#             for i in list_interface:
#                 count += 1
#                 i_interface = f'{count} - {i.capitalize()}'
#                 list_response.append(i_interface)
#             response = list_response
#             interface = False
#             model = False
#             problem = False
#             return response, interface, model, problem,first_question
#     else:
    response, model, problem,first_question = get_model(input_user,subject,df_subject,device,df_device,interface,first_question)
    return response, interface, model, problem,first_question

def get_model(input_user,subject,df_subject,device,df_device,interface,first_question):
    # modelo
    df_interface = df_device.loc[df_device['interface'] == interface]
    troubleshooting4 = df_device.groupby('model')
    list_model = [i for i,j in troubleshooting4]
    vetor = 0
    linha = None
    for i in list_model:
        list_model_df = df_interface.loc[df_interface['model']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_model_df)
        if vetor_encontrado > vetor:
            vetor = vetor_encontrado
            linha = list_model_df.iloc[indice_sentenca]
    model = linha['model']
    # if vetor < 0.7:
    #     # perguntas
    #     df_questions = pd.read_excel(f'{pai_path}\\database\\question.xlsx')
    #     list_response = []
    #     if first_question:
    #         first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
    #         list_response.append(first_question)
    #         first_question = False
    #     if vetor > 0.3:
    #         df_moment1 = df_questions.loc[df_questions['subject'] == subject]
    #         df_moment2 = df_moment1.loc[df_moment1['device'] == device]
    #         df_moment3 = df_moment2.loc[df_moment2['interface'] == interface]
    #         df_moment4 = df_moment3.loc[df_moment3['model'] == model]
    #         linha = df_moment4.iloc[indice_sentenca]
    #         list_response.append(linha['model_question'])
    #         response = list_response
    #         model = False
    #         problem = False
    #         return response, model, problem,first_question
    #     else:
    #         intro = "Qual dessas opções o seu problema melhor se enquadra?"
    #         list_response.append(intro)
    #         count = 0
    #         for i in list_model:
    #             count += 1
    #             i_model = f'{count} - {i.capitalize()}'
    #             list_response.append(i_model)
    #         response = list_response
    #         model = False
    #         problem = False
    #         return response, model, problem,first_question
    # else:
    response,problem,first_question = get_problem(input_user,subject,device,interface,df_interface,model,first_question)
    return response, model, problem,first_question

def get_problem(input_user,subject,device,interface,df_interface,model,first_question):
    # problem
    df_model = df_interface.loc[df_interface['model'] == model]
    troubleshooting5 = df_model.groupby('problem')
    list_problem = [i for i,j in troubleshooting5]
    vetor = 0
    linha = None
    for i in list_problem:
        list_problem_df = df_model.loc[df_model['problem']==i]
        vetor_encontrado, indice_sentenca = tf_idf(input_user, list_problem_df)
        if vetor_encontrado > vetor:
            vetor = vetor_encontrado
            linha = list_problem_df.iloc[indice_sentenca]
    problem = linha['problem']
    # if vetor < 0.7:
    #     # perguntas
    #     df_questions = pd.read_excel(f'{pai_path}\\database\\question.xlsx')
    #     list_response = []
    #     if first_question:
    #         first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
    #         list_response.append(first_question)
    #         first_question = False
    #     if vetor > 0.3:
    #         df_moment1 = df_questions.loc[df_questions['subject'] == subject]
    #         df_moment2 = df_moment1.loc[df_moment1['device'] == device]
    #         df_moment3 = df_moment2.loc[df_moment2['interface'] == interface]
    #         df_moment4 = df_moment3.loc[df_moment3['model'] == model]
    #         df_moment5 = df_moment4.loc[df_moment4['problem'] == problem]
    #         linha = df_moment5.iloc[indice_sentenca]
    #         list_response.append(linha['problem_question'])
    #         response = list_response
    #         problem = False
    #         return response, problem,first_question
    #     else:
    #         intro = "Qual dessas opções o seu problema melhor se enquadra?"
    #         list_response.append(intro)
    #         count = 0
    #         for i in list_problem:
    #             count += 1
    #             i_problem = f'{count} - {i.capitalize()}'
    #             list_response.append(i_problem)
    #         response = list_response
    #         problem = False
    #         return response, problem,first_question
    # else:
    response = problem
    return response, problem,first_question