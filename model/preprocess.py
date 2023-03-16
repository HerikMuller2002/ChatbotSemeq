from re import sub
from numpy import delete
from random import choice
from spacy import load
from nltk import download
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer # pip install scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

class Tratamento:
    # Função para pré-processar os textos para calculo de correlação
    def preprocess(intents,lematizar=True):
        lemmatizer = WordNetLemmatizer()
        # inicializaremos nossa lista de palavras, classes, documentos e 
        # definimos quais palavras serão ignoradas
        words = []
        documents = []
        download('punkt')
        nlp = load("pt_core_news_sm") # python -m spacy download pt_core_news_sm => erro de linguagem
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                word = pattern.lower()
                word = sub(r"[!#$%&'()*+,-./:;<=>?@[^_`{|}~]+", ' ', sub('[áàãâä]', 'a', sub('[éèêë]', 'e', sub('[íìîï]', 'i', sub('[óòõôö]', 'o', sub('[úùûü]', 'u', word))))))
                # tirar espaços em branco
                word = sub(r'\s+', ' ',word)
                # com ajuda no nltk fazemos aqui a tokenizaçao dos patterns 
                # e adicionamos na lista de palavras
                word = word_tokenize(word)
                words.extend(word)
                # adiciona aos documentos para identificarmos a tag para a mesma
                documents.append((word, intent['tag']))
        if lematizar:
            words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
        # tirar pontuações, acentos e espaços extras
        return word.strip()
    
    def get_values_vetor(lista,similaridade):
        # Filtra a lista para manter apenas os valores acima de 40% de relação
        lista_filtrada = [x for x in lista if x > 0.4]
        # Classifica a lista filtrada em ordem decrescente
        sorted_lista = sorted(lista_filtrada, reverse=True)
        # Seleciona no máximo três valores acima de zero
        lista = sorted_lista[:3]
        value_counts = {}
        for value in lista:
            if value not in value_counts:
                value_counts[value] = 1
            else:
                value_counts[value] += 1
        unique_values = sorted(set(lista), reverse=True)
        values_vetor = []
        for value in unique_values:
            count = value_counts[value]
            if len(values_vetor) < 3:
                values_vetor.extend([value] * min(count, 3 - len(values_vetor)))
            elif count > 1:
                indices = [i for i, x in enumerate(lista) if x == value]
                chosen_index = choice(indices)
                values_vetor[chosen_index % 3] = value
        indices_sentenca = []
        if len(lista_filtrada) > 0:
            count = -2
            for vetor in lista_filtrada:
                indices_sentenca.append(similaridade.argsort()[0][count])
                count -= 1
        else:
            pass
        return values_vetor, indices_sentenca

    # Função para calcular correlação
    def tf_idf(user_input,list_text_db):
        list_text_db.append(user_input)
        tfidf = TfidfVectorizer()
        list_text_db = list(filter(None, list_text_db))
        palavras_vetorizadas = tfidf.fit_transform(list_text_db)
        similaridade = cosine_similarity(palavras_vetorizadas[-1], palavras_vetorizadas)
        vetor_similar = similaridade.flatten()
        vetor_similar.sort()
        vetor_similar = delete(vetor_similar, -1)
        vetor_encontrado = Tratamento.get_values_vetor(vetor_similar,similaridade)
        indices_sentenca = []
        if len(vetor_encontrado) > 0:
            count = -2
            for vetor in vetor_encontrado:
                indices_sentenca.append(similaridade.argsort()[0][count])
                count -= 1
        else:
            pass
        return vetor_encontrado,indices_sentenca