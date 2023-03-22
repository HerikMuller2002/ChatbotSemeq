import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from re import sub

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

def get_subject(input_user, first_question):
    # assunto
    df = pd.read_excel('\\Users\\Semeq\\Desktop\\ChatbotSemeq\\database\\troubleshooting.xlsx')
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
    if vetor_encontrado < 0.7:
        # perguntas
        df_questions = pd.read_excel('\\Users\\Semeq\\Desktop\\ChatbotSemeq\\database\\question.xlsx')
        first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
        list_response = []
        if first_question:
            list_response.append(first_question)
        if vetor > 0.2:
            df_moment1 = df_questions.loc[df_questions['subject'] == subject]
            linha = df_moment1.iloc[indice_sentenca]
            response = linha['subject_question']
            return [response]
        else:
            intro = "Qual dessas opções o seu problema melhor se enquadra?"
            list_response.append(intro)
            count = 0
            for i in list_subject:
                count += 1
                i_subject = f'{count} - {i.capitalize()}'
                list_response.append(i_subject)
            response = list_response
        return response
    else:
        return get_device(input_user,df,subject,first_question)

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
    if vetor_encontrado < 0.7:
        # perguntas
        df_questions = pd.read_excel('\\Users\\Semeq\\Desktop\\ChatbotSemeq\\database\\question.xlsx')
        first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
        list_response = []
        if first_question:
            list_response.append(first_question)
        if vetor > 0.2:
            df_moment1 = df_questions.loc[df_questions['subject'] == subject]
            df_moment2 = df_moment1.loc[df_moment1['device'] == device]
            linha =  df_moment2.iloc[indice_sentenca]
            response = linha['device_question']
            return [response]
        else:
            intro = "Qual dessas opções o seu problema melhor se enquadra?"
            list_response.append(intro)
            count = 0
            for i in list_device:
                count += 1
                i_device = f'{count} - {i.capitalize()}'
                list_response.append(i_device)
            response = list_response
        return response
    else:
        return get_interface(input_user,df,subject,df_subject,device,first_question)

def get_interface(input_user,df,subject,df_subject,device,first_question):
    # interface
    df_device = df_subject.loc[df['device'] == device]
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
    if vetor_encontrado < 0.7:
        # perguntas
        df_questions = pd.read_excel('\\Users\\Semeq\\Desktop\\ChatbotSemeq\\database\\question.xlsx')
        first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
        list_response = []
        if first_question:
            list_response.append(first_question)
        if vetor > 0.2:
            df_moment1 = df_questions.loc[df_questions['subject'] == subject]
            df_moment2 = df_moment1.loc[df_moment1['device'] == device]
            df_moment3 = df_moment2.loc[df_moment2['interface'] == interface]
            linha = df_moment3.iloc[indice_sentenca]
            response = linha['interface_question']
            interface = False
            model = False
            problem = False
            return response, interface, model, problem
        else:
            intro = "Qual dessas opções o seu problema melhor se enquadra?"
            list_response.append(intro)
            count = 0
            for i in list_interface:
                count += 1
                i_interface = f'{count} - {i.capitalize()}'
                list_response.append(i_interface)
            response = list_response
            interface = False
            model = False
            problem = False
            return response, interface, model, problem
    else:
        response, model, problem = get_model(input_user,df,subject,df_subject,device,df_device,interface,first_question)
        return response, interface, model, problem

def get_model(input_user,df,subject,df_subject,device,df_device,interface,first_question):
    # modelo
    df_interface = df_device.loc[df['interface'] == interface]
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
    if vetor_encontrado < 0.7:
        # perguntas
        df_questions = pd.read_excel('\\Users\\Semeq\\Desktop\\ChatbotSemeq\\database\\question.xlsx')
        first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
        list_response = []
        if first_question:
            list_response.append(first_question)
        if vetor > 0.2:
            df_moment1 = df_questions.loc[df_questions['subject'] == subject]
            df_moment2 = df_moment1.loc[df_moment1['device'] == device]
            df_moment3 = df_moment2.loc[df_moment2['interface'] == interface]
            df_moment4 = df_moment3.loc[df_moment3['model'] == model]
            linha = df_moment3.iloc[indice_sentenca]
            response = linha['model_question']
            model = False
            problem = False
            return response, model, problem
        else:
            intro = "Qual dessas opções o seu problema melhor se enquadra?"
            list_response.append(intro)
            count = 0
            for i in list_model:
                count += 1
                i_model = f'{count} - {i.capitalize()}'
                list_response.append(i_model)
            response = list_response
            model = False
            problem = False
            return response, model, problem
    else:
        response,problem = get_problem(input_user,df,subject,df_subject,device,df_device,interface,df_interface,model,first_question)
        return response, model, problem

def get_problem(input_user,df,subject,df_subject,device,df_device,interface,df_interface,model,first_question):
    # problem
    df_model = df_interface.loc[df['model'] == model]
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
    if vetor_encontrado < 0.7:
        # perguntas
        df_questions = pd.read_excel('\\Users\\Semeq\\Desktop\\ChatbotSemeq\\database\\question.xlsx')
        first_question = "Estou com dificuldade de identificar seu problema na base de dados! Irei fazer algumas perguntas para que eu possa entender melhor ok!?"
        list_response = []
        if first_question:
            list_response.append(first_question)
        if vetor > 0.2:
            df_moment1 = df_questions.loc[df_questions['subject'] == subject]
            df_moment2 = df_moment1.loc[df_moment1['device'] == device]
            df_moment3 = df_moment2.loc[df_moment2['interface'] == interface]
            df_moment4 = df_moment3.loc[df_moment3['model'] == model]
            df_moment5 = df_moment4.loc[df_moment4['problem'] == problem]
            linha = df_moment3.iloc[indice_sentenca]
            response = linha['problem_question']
            problem = False
            return response, problem
        else:
            intro = "Qual dessas opções o seu problema melhor se enquadra?"
            list_response.append(intro)
            count = 0
            for i in list_problem:
                count += 1
                i_problem = f'{count} - {i.capitalize()}'
                list_response.append(i_problem)
            response = list_response
            problem = False
            return response, problem
    else:
        response = False
        return response, problem