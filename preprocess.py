from re import sub
from numpy import delete
from random import choice
from spacy import load
from nltk import download
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer # pip install scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

class Tratamento:
    # Função para pré-processar os textos para calculo de correlação
    def preprocess_model(text):
        words = []
        documents = []
        if type(text) == dict:
            for intent in text['intents']:
                for pattern in intent['patterns']:
                    pattern = pattern.lower()
                    pattern = sub(r"[!#$%&'()*+,-./:;<=>?@[^_`{|}~]+", ' ', sub('[áàãâä]', 'a', sub('[éèêë]', 'e', sub('[íìîï]', 'i', sub('[óòõôö]', 'o', sub('[úùûü]', 'u', pattern))))))
                    # tirar espaços em branco
                    pattern = sub(r'\s+', ' ',pattern)
                    # com ajuda no nltk fazemos aqui a tokenizaçao dos patterns 
                    # e adicionamos na lista de palavras
                    pattern = word_tokenize(pattern)
                    words.extend(pattern)
                    # lematizamos as palavras ignorando os palavras da lista ignore_words
                    words = [WordNetLemmatizer().lemmatize(w.lower())for w in words]
                    # adiciona aos documentos para identificarmos a tag para a mesma
                    documents.append((pattern, intent['tag']))
        return words,documents
    
    def preprocess_lemma(text):
        nlp = load("pt_core_news_sm")
        # encontrar radical das palavras (lematização)
        documento = nlp(text)
        text = []
        for token in documento:
            text.append(token.lemma_)
        text = ' '.join([str(elemento) for elemento in text if not elemento.isdigit()])
        return text
    
    def preprocess_input(text):
        text = text.lower().strip()
        # tirar pontuações, acentos e espaços extras
        text = sub(r"[!#$%&'()*+,-./:;<=>?@[^_`{|}~]+", ' ', sub('[áàãâä]', 'a', sub('[éèêë]', 'e', sub('[íìîï]', 'i', sub('[óòõôö]', 'o', sub('[úùûü]', 'u', text))))))
        # tirar espaços em branco
        text = sub(r'\s+', ' ',text)
        return text
    
    def correcao(text):
        list_text = text.split()
        spell = SpellChecker(language='pt')
        # Criar uma nova lista com as correções ortográficas
        correcoes = [spell.correction(palavra) for palavra in list_text]
        correcoes = [c for c in correcoes if c is not None]
        if len(correcoes) > 1:
            # Juntar a lista corrigida em uma única string
            text = ' '.join(correcoes)
        else:
            pass
        return text

class Correlacao():
    def get_values_vetor(lista):
        # Filtra a lista para manter apenas os valores acima de zero
        lista_filtrada = [x for x in lista if x > 0]
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
        return values_vetor

    # Função para calcular correlação
    # def tf_idf(user_input,list_text_db):
    #     list_text_db.append(user_input)
    #     tfidf = TfidfVectorizer()
    #     list_text_db = list(filter(None, list_text_db))
    #     palavras_vetorizadas = tfidf.fit_transform(list_text_db)
    #     similaridade = cosine_similarity(palavras_vetorizadas[-1], palavras_vetorizadas)
    #     vetor_similar = similaridade.flatten()
    #     vetor_similar.sort()
    #     vetor_similar = delete(vetor_similar, -1)
    #     vetor_encontrado = Tratamento.get_values_vetor(vetor_similar)
    #     count = -2
    #     indices_sentenca = []
    #     for i in vetor_encontrado:
    #         indices_sentenca.append(similaridade.argsort()[0][count])
    #         count -= 1
    #     return vetor_encontrado,indices_sentenca

    def tf_idf(user_input,list_text_db):
        list_text_db.append(user_input)
        tfidf = TfidfVectorizer()
        list_text_db = list(filter(None, list_text_db))
        palavras_vetorizadas = tfidf.fit_transform(list_text_db)
        similaridade = cosine_similarity(palavras_vetorizadas[-1], palavras_vetorizadas)
        vetor_similar = similaridade.flatten()
        vetor_similar.sort()
        vetor_similar = delete(vetor_similar, -1)
        vetor_encontrado = vetor_similar[-1]
        indices_sentenca = similaridade.argsort()[0][-2]
        return vetor_encontrado,indices_sentenca