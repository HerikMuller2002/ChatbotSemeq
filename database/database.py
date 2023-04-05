import sqlite3
import json
import pandas as pd

def add_json_to_db(json_data, db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    for intent in json_data:
        # Insere a tag na tabela intents
        tag = intent['tag']
        description = '' # Neste exemplo, não há descrição, mas poderia ser adicionada
        c.execute("INSERT INTO intents (tag, description) VALUES (?, ?)", (tag, description))
        intent_id = c.lastrowid

        # Insere os padrões na tabela patterns
        for pattern in intent['patterns']:
            c.execute("INSERT INTO patterns (intent_id, pattern) VALUES (?, ?)", (intent_id, pattern))

        # Insere as respostas na tabela responses
        for response in intent['responses']:
            c.execute("INSERT INTO responses (intent_id, response) VALUES (?, ?)", (intent_id, response))

    conn.commit()
    conn.close()

  
def add_df_to_db(df, db_file):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    for id,row in df.iterrows():
        # Insere a tag na tabela intents
        c.execute("INSERT INTO problems (subject, device, interface, model, problem, description) VALUES (?, ?, ?, ?, ?, ?)", (row['subject'], row['device'], row['interface'], row['model'], row['problem'], row['description']))
    conn.commit()
    conn.close()

df = pd.read_excel('database\\troubleshooting.xlsx')
df = df.drop('solution',axis=1)

# add_df_to_db(df,'database\\chatbot.db')