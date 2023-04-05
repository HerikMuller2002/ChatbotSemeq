import pandas as pd
import numpy as np

from re import sub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.util import ngrams

def preprocess_input(text):
    text = text.lower().strip()
    # tirar pontuações, acentos e espaços extras
    text = sub('[áàãâä]', 'a', sub('[éèêë]', 'e', sub('[íìîï]', 'i', sub('[óòõôö]', 'o', sub('[úùûü]', 'u', text)))))
    # tirar espaços em branco
    text = sub(r'\s+', ' ',text)
    return text

#########################################################

def preprocess_list(a, b):
    new_list = []
    new_copy_list = []
    for i, text in enumerate(a):
        text = preprocess_input(text)
        if ',' in text or '\\' in text or '/' in text:
            new_texts = text.split(',') + text.split('\\') + text.split('/')
            for new_text in new_texts:
                new_list.append(new_text)
                new_copy_list.append({"value": new_text, "line": b[i]["line"], "column": b[i]["column"]})
        else:
            new_texts = text.split()
            for new_text in new_texts:
                new_list.append(new_text)
                new_copy_list.append({"value": new_text, "line": b[i]["line"], "column": b[i]["column"]})
            new_list.append(text)
            new_copy_list.append({"value": text, "line": b[i]["line"], "column": b[i]["column"]})
    return new_list, new_copy_list

def preprocess_nrange(list_text_db, nrange=1):
    preprocessed_list = []
    for text in list_text_db:
        tokens = word_tokenize(text.lower())
        if nrange > 1:
            tokens = [" ".join(ng) for ng in ngrams(tokens, nrange)]
        preprocessed_list.append(" ".join(tokens))
    return preprocessed_list

# função para calcular tf e idf (similaridade entre um input com os dados do banco de dados)
def tf_idf(user_input, dados, num_vetores=1, nrange=1):
    # verifica se os dados passados como parâmetros na base de dados são um dataframe
    if isinstance(dados, pd.DataFrame):
        # converte os dados do dataframe em string
        dataframe = dados.astype(str)
        # transforma o dataframe em lista
        list_text_db = dataframe.to_numpy().flatten().tolist()

        # loop para criar uma lista "cópia" com dicionários, com os seus respectivos elementos e indices do dataframe
        # para consulta!
        list_text_db_copy = []
        for i, row in dados.iterrows():
            for col in row.index:
                value = str(row[col])
                list_text_db_copy.append({'value': value, 'line': i, 'column': col})

    # verifica se os dados passados como parâmetros na base de dados são uma lista
    elif isinstance(dados, list):
        list_text_db = dados.copy()
        # loop para criar uma lista "cópia" com dicionários, com os seus respectivos elementos e indices do dataframe
        # para consulta!
        list_text_db_copy = []
        for i, value in enumerate(dados):
            list_text_db_copy.append({'value': value, 'line': i})

    # passa as listas criadas na função preprocess_list para que sejam splitadas, aumentando a quantidade de palavras nas listas
    list_text_db,list_text_db_copy = preprocess_list(list_text_db,list_text_db_copy)

    # loop para eliminar elementos duplicados nos mesmos indices
    unicos = {}
    for i in list_text_db_copy:
        key = i['value'] + '-' + str(i['line'])
        if key not in unicos:
            unicos[key] = i
        else:
            id = list_text_db_copy.index(i)
            list_text_db.pop(id)
            list_text_db_copy.remove(i)

    # pré-processando a lista original para "ngramas" passados pelo parâmetro "nrange" da função.
    # ngramas define a quantidade de colunas na matriz que será calculada no tf e idf, matriz: MxN -> M linhas x N colunas => M é a quantidade de combinações possíveis com as palavras das frases, de acordo com ngramas; e N é ngramas
    list_text_db = preprocess_nrange(list_text_db, nrange)
    # adiciona o input do usuário como último elemento na matriz, para servir como parâmetro = 1 no cálculo
    list_text_db.append(user_input)

    # transforma a matriz em vetor e calculando a similaridade
    tfidf = TfidfVectorizer()
    palavras_vetorizadas = tfidf.fit_transform(list_text_db)
    similaridade = cosine_similarity(palavras_vetorizadas[-1], palavras_vetorizadas)
    vetor_similar = similaridade.flatten()

    # determina os índices dos vetores na lista list_text_db
    indices = np.argsort(similaridade, axis=1)
    indices = np.flip(indices, axis=1)
    indices = indices[:, 1:num_vetores+1]

    # determina a lista de vetores encontrados
    vetores = [vetor_similar[indices[0][i]] for i in range(num_vetores)]
    # determina a lista dos indices dos vetores encotrados
    indices_sentencas = [int(indices[0][i]) for i in range(num_vetores)]
    
    # loop para criar lista resultados com dicionários, com a quantidade de elementos definido no parâmetro da função
    resultados = []
    for i in range(num_vetores):
        resultado = {"vetor": vetores[i], "indice": list_text_db_copy[indices[0][i]]['line'], "valor": list_text_db[indices_sentencas[i]]}
        resultados.append(resultado)

    # retorna a lista resultados com os dicionários dos vetores e indices encontrados
    return resultados