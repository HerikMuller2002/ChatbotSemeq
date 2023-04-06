from spellchecker import SpellChecker
from spacy import load
from re import sub
from nltk import word_tokenize
from nltk.util import ngrams
from spacy.lang.pt.stop_words import STOP_WORDS
from nltk.stem.snowball import SnowballStemmer

nlp = load("pt_core_news_sm")

def preprocess_correcao(text):
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

def preprocess_lemma(text):
    # encontrar radical das palavras (lematização)
    doc = nlp(text)
    lemmas = [token.lemma_ if token.pos_ not in ["PUNCT"] and token.text not in STOP_WORDS else token.text for token in doc]
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

def preprocess_input(text):
    text = text.lower().strip()
    # tirar pontuações, acentos e espaços extras
    text = sub('[áàãâä]', 'a', sub('[éèêë]', 'e', sub('[íìîï]', 'i', sub('[óòõôö]', 'o', sub('[úùûü]', 'u', text)))))
    # tirar espaços em branco
    text = sub(r'\s+', ' ',text)
    return text

def preprocess_list(list_text_db, list_text_db_copy):
    new_list = []
    new_copy_list = []
    for i, text in enumerate(list_text_db):
        text = preprocess_input(text)
        if ',' in text or '\\' in text or '/' in text:
            new_texts = text.split(',') + text.split('\\') + text.split('/')
            for new_text in new_texts:
                new_list.append(new_text)
                new_copy_list.append({"value": new_text, "line": list_text_db_copy[i]["line"], "column": list_text_db_copy[i]["column"]})
        else:
            new_texts = text.split()
            for new_text in new_texts:
                new_list.append(new_text)
                new_copy_list.append({"value": new_text, "line": list_text_db_copy[i]["line"], "column": list_text_db_copy[i]["column"]})
            new_list.append(text)
            new_copy_list.append({"value": text, "line": list_text_db_copy[i]["line"], "column": list_text_db_copy[i]["column"]})
    return new_list, new_copy_list

def preprocess_nrange(list_text_db, nrange=1):
    preprocessed_list = []
    for text in list_text_db:
        tokens = word_tokenize(text.lower())
        if nrange > 1:
            tokens = [" ".join(ng) for ng in ngrams(tokens, nrange)]
        preprocessed_list.append(" ".join(tokens))
    return preprocessed_list