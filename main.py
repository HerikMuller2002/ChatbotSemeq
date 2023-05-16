from services.preproccess import *
from services.extract import class_prediction, get_response
from services.database import *
from services.similarity import *

df_responses = read_db('responses')
df_intents = read_db('intents')
df_patterns = read_db('patterns')
df_censored = df_patterns.loc[df_patterns['intent_id'] == 1]

def chatbot_run(input_user):
    # input do usuário é pré-processado
    input_user = preprocess_input(input_user)
    vetores = []
    for i in range(len(input_user.split())+1):
        vetores.append({'valor':tf_idf(input_user,df_censored,1,i)[0]['vetor'],'peso':i})
    for j in range(len(vetores)):
        vetores[j]['vetor'] = vetores[j]['vetor']*vetores[j]['peso']

    soma_num = vetores.sum()
    soma_peso = 0
    for y in vetores:
        soma_peso += y['peso']
    # model_path = "model\\model.h5"
    # words_path = "model\\words.pkl"
    # classes_path = "model\\classes.pkl"

    # intent_user = class_prediction(input_user, model_path,words_path,classes_path,0.3)
    # response = get_response(intent_user, df_responses, df_intents)

    # return response

a = chatbot_run("olá, tenha um bom dia")