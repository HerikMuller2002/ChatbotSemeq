import os
import json
# from preprocess import Tratamento

# with open(('logs\\log.json'), 'r', encoding='utf-8') as log_chat:
#     log = json.load(log_chat)
# original_dict = log[-1]
# new_dict = {
#     "subject": original_dict["subject"],
#     "device": original_dict["device"],
#     "interface": original_dict["interface"],
#     "model": original_dict["model"],
#     "problem": original_dict["problem"]
#     }
# for chave, valor in new_dict.items():
#     if isinstance(valor, bool):
#         subcontext = chave
#         value_subcontext = valor
#         break
# print(subcontext)

with open("intents.json",'r',encoding="UTF-8") as banco:
    intents = json.load(banco)

# words,documents = Tratamento.preprocess_model(intents)
# print(words)
# print()
# print(documents)

# a = Tratamento.preprocess_lemma("Por exemplo, digamos que você queira enriquecer os logs com informações adicionais, como o endereço IP do usuário que fez a pergunta, a data e hora em que a pergunta foi feita, ou informações do agente do usuário. Com o Logstash, você pode usar plugins de entrada, como o plugin HTTP, para capturar informações adicionais, e plugins de filtragem, como o plugin GeoIP, para enriquecer os dados.")
# print(a)

import logging

# Configura o logger
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

def registra_log(assunto, pergunta, resposta, contexto):
    logging.info(f'Assunto atual: {assunto}')
    logging.info(f'Pergunta: {pergunta}')
    logging.info(f'Resposta: {resposta}')
    logging.info(f'Contexto: {contexto}')

registra_log("1","1","1","1")