import os
import logging

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))

def log(**kwargs):
    # Configura o logger
    logging.basicConfig(filename= os.path.join(path, 'chatbot.log'), level=logging.INFO)
    # Verifica se o arquivo de log existe
    if os.path.exists(os.path.join(path, 'chatbot.log')):
        # Abre o arquivo de log no modo leitura e gravação
        with open(os.path.join(path, 'chatbot.log'), 'r+') as f:
            # Move o cursor para o final do arquivo
            f.seek(0, os.SEEK_END)
            # Verifica se o tamanho do arquivo é diferente de zero
            if f.tell() != 0:
                # Adiciona uma linha em branco antes de adicionar o novo registro de log
                f.write('\n')
    # Adiciona o novo registro de log
    for chave, valor in kwargs.items():        
        logging.info(f'{chave}: {valor}')


def clear_log():
    try:
        os.remove(os.path.join(path, 'chatbot.log'))
    except FileNotFoundError:
        pass


def get_log(chave):
    with open(os.path.join(path, 'chatbot.log'), 'r') as f:
        linhas = f.readlines()
        for linha in linhas:
            if chave in linha:
                return linha.split(": ")[1].strip()