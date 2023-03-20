import sqlite3
import json

def criar_db():
    # estabelece uma conexão com o banco de dados
    conexao = sqlite3.connect('database.db')
    # cria uma tabela chamada 'mensagens' com duas colunas: 'mensagem' e 'resposta'
    cursor = conexao.cursor()
    #cursor.execute('CREATE TABLE problemas (assunto, app_device, interface, modelo, problema, solucao)')
    cursor.execute('CREATE TABLE solution (subject, device, interface, model, problem, solution)')
    cursor.execute('CREATE TABLE intent_casual (intent, pattern, response)')
    cursor.execute('CREATE TABLE intent_censored ()')
    cursor.execute('CREATE TABLE intent_subject ()')
    cursor.execute('CREATE TABLE intent_device ()')
    cursor.execute('CREATE TABLE intent_interface ()')
    cursor.execute('CREATE TABLE intent_model ()')
    cursor.execute('CREATE TABLE intent_problem ()')
    conexao.commit()
    # fecha a conexão com o banco de dados
    conexao.close()