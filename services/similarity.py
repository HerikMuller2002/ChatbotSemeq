import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from services.preprocess import preprocess_list,preprocess_nrange

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
        a = 0
        for i, row in dados.iterrows():
            a += 1
            print(a)
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
            print(3)
            list_text_db_copy.append({'value': value, 'line': i})
    # passa as listas criadas na função preprocess_list para que sejam splitadas, aumentando a quantidade de palavras nas listas
    list_text_db,list_text_db_copy = preprocess_list(list_text_db,list_text_db_copy)
    # loop para eliminar elementos duplicados nos mesmos indices
    unicos = {}
    for i in list_text_db_copy:
        print(4)
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
        print(5)
        resultado = {"vetor": vetores[i], "indice": list_text_db_copy[indices[0][i]]['line'], "valor": list_text_db[indices_sentencas[i]]}
        resultados.append(resultado)
    # retorna a lista resultados com os dicionários dos vetores e indices encontrados
    return resultados