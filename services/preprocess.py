import os
import sys
# Obtenha o diretório pai do arquivo atual
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from spellchecker import SpellChecker
from spacy import load
from re import sub
from nltk import word_tokenize
from nltk.util import ngrams
from spacy.lang.pt.stop_words import STOP_WORDS
from nltk.stem.snowball import SnowballStemmer
from services.database import *

from numba import jit



nlp = load("pt_core_news_sm")

def preprocess_input(text):
    text = sub(r"[!#$%&'()*+,-./:;<=>?@[^_`{|}~]+", ' ',text)
    text = sub(r'\s+', ' ',text)
    text = preprocess_correcao(text)
    text = text.lower().strip()
    # tirar pontuações, acentos e espaços extras
    text = sub('[áàãâä]', 'a', sub('[éèêë]', 'e', sub('[íìîï]', 'i', sub('[óòõôö]', 'o', sub('[úùûü]', 'u', text)))))
    # tirar espaços em branco
    text = sub(r'\s+', ' ',text)
    return text


def preprocess_correcao(text):
    list_input = text.split()
    list_text = []
    spell = SpellChecker(language='pt')
    df_patterns = set(read_db('patterns').drop(['id','intent_id'], axis=1).to_numpy().flatten().tolist())
    df_problems = set(read_db('problems').drop(['id','problem','description'], axis=1).to_numpy().flatten().tolist())
    for i in list_input:
        if i not in STOP_WORDS and (i in df_problems or i in df_patterns):
            list_text.append(i)
        else:
            # Cria uma nova lista com as correções ortográficas
            correcao = spell.correction(i)
            list_text.append(correcao)
    try:
        text = ' '.join(list_text)
    except TypeError:
        text = ' '.join(list_input)
    return text


def preprocess_lemma(text):
    # encontrar radical das palavras (lematização)
    doc = nlp(text)
    lemmas = [token.lemma_ if token.text not in STOP_WORDS else token.text for token in doc]
    return " ".join(lemmas)


def preprocess_stem(text):
    stemmer = SnowballStemmer("portuguese")
    tokens = word_tokenize(text)
    stems = [stemmer.stem(token) for token in tokens]
    text = ' '.join([str(element) for element in stems])
    return text


def preprocess_model(df_patterns, df_intents):
    words = []
    documents = []
    count = 0
    for group_name, group_df in df_patterns.groupby("intent_id"):
        for index, row in group_df.iterrows():
            count += 1
            print("linha:",count,"de:",df_patterns.shape[0])
            pattern = row['pattern'].lower()
            pattern = sub(r"[!#$%&'()*+,-./:;<=>?@[^_`{|}~]+", ' ', sub('[áàãâä]', 'a', sub('[éèêë]', 'e', sub('[íìîï]', 'i', sub('[óòõôö]', 'o', sub('[úùûü]', 'u', pattern))))))
            pattern = sub(r'\s+', ' ',pattern)
            pattern = word_tokenize(pattern)
            words.extend(pattern)
            words2 = [preprocess_lemma(w).lower() for w in words]
            intent = df_intents.loc[df_intents['id'] == group_name, 'tag'].values[0]
            documents.append((pattern, intent))
    return words2, documents


# def preprocess_list(list_text_db, list_text_db_copy):
#     new_list = []
#     new_copy_list = []
#     for i, text in enumerate(list_text_db):
#         text = preprocess_input(text)
#         if ',' in text or '\\' in text or '/' in text:
#             new_texts = text.split(',') + text.split('\\') + text.split('/')
#             for new_text in new_texts:
#                 new_list.append(new_text)
#                 new_copy_list.append({"value": new_text, "line": list_text_db_copy[i]["line"], "column": list_text_db_copy[i]["column"]})
#         else:
#             new_texts = text.split()
#             for new_text in new_texts:
#                 new_list.append(new_text)
#                 new_copy_list.append({"value": new_text, "line": list_text_db_copy[i]["line"], "column": list_text_db_copy[i]["column"]})
#             new_list.append(text)
#             new_copy_list.append({"value": text, "line": list_text_db_copy[i]["line"], "column": list_text_db_copy[i]["column"]})
#     return new_list, new_copy_list


def preprocess_nrange(list_text_db, nrange=1):
    preprocessed_list = []
    for text in list_text_db:
        tokens = word_tokenize(text.lower())
        if nrange > 1:
            tokens = [" ".join(ng) for ng in ngrams(tokens, nrange)]
        preprocessed_list.append(" ".join(tokens))
    return preprocessed_list





def preprocess_list(list_text_db):
    new_list = []
    new_copy_list = []
    for i, text in enumerate(list_text_db):
        text = preprocess_input(text)
        if ',' in text or '\\' in text or '/' in text or len(text) > 1:
            new_texts = text.split(',') + text.split('\\') + text.split('/') + text.split()
            for new_text in new_texts:
                new_list.append(new_text)
        else:
            new_list.append(text)
    return new_list

valores_masculinos = ['primeiro', 'segundo', 'terceiro', 'quarto', 'quinto', 'sexto', 'sétimo', 'oitavo', 'nono', 'décimo']
valores_femininos = ['primeira', 'segunda', 'terceira', 'quarta', 'quinta', 'sexta', 'sétima', 'oitava', 'nona', 'décima']
ordinais = valores_femininos + valores_masculinos
numeros_escritos = ['um', 'dois', 'três', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove', 'dez', 'onze', 'doze', 'treze', 'catorze', 'quinze', 'dezesseis', 'dezessete', 'dezoito', 'dezenove', 'vinte']

df = pd.read_excel(f'{parent_dir}\\database\\troubleshooting.xlsx')
dataframe = df.astype(str)
# transforma o dataframe em lista
list_text_db = dataframe.to_numpy().flatten().tolist()
list_text_db1 = preprocess_list(list_text_db)
print(list_text_db1)

STOP_WORDS = [word for word in STOP_WORDS if not word.isdigit() and word not in ordinais and word not in numeros_escritos and word not in list_text_db]
# print(STOP_WORDS)
# print('sistema' in list_text_db)