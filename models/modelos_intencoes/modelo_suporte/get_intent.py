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

def get_solution(input_user):

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

    # interface
    df_subject = df.loc[df['subject'] == subject]
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

    # modelo
    df_subject = df.loc[df['subject'] == subject]
    df_device = df_subject.loc[df['device'] == device]
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

    # problem
    df_subject = df.loc[df['subject'] == subject]
    df_device = df_subject.loc[df['device'] == device]
    df_interface = df_device.loc[df['interface'] == interface]
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

    response = f"{subject} / {device} / {interface} / {model} / {problem}"
    return response
    
# filtro_subject = df.loc[df['subject'] == subject]
# lista_subject = filtro_subject.values.tolist()

# filtro_device = filtro_subject.loc[filtro_subject['device'] == device]


# filtro_interface = filtro_device.loc[filtro_device['interface'] == interface]


# filtro_model = df.loc[df['subject'] == model]
# lista_model = filtro_model.values.tolist()
