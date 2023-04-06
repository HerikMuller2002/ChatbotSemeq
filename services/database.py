import pandas as pd
import sqlite3
import os

path_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))

def read_db(table_name):
  # Conectar com o banco de dados
  db_file = os.path.join(path_db, 'chatbot.db')
  conn = sqlite3.connect(db_file)
  # Ler a tabela e transformar em um dataframe pandas
  df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
  # Fechar a conexão com o banco de dados
  conn.close()
  # Retornar o dataframe
  return df