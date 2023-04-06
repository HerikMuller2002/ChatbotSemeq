import os
import logging

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))

def log(**kwargs):
    # Configura o logger
    logging.basicConfig(filename= os.path.join(path, 'chatbot.log'), level=logging.INFO)
    for chave, valor in kwargs.items():        
        logging.info(f'{chave}: {valor}')
        
def clear_log():
    try:
        os.remove(os.path.join(path, 'chatbot.log'))
    except FileNotFoundError:
        pass