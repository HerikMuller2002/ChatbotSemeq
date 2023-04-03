import sqlite3
import json

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

json_data = [
      {
        "tag": "welcome",
        "patterns": ["oi", "bom dia", "boa tarde", "boa noite", "good morning", "hi", "hello", "olá", "ola", "hey", "eai", "eae", "e ai", "e ae", "esta bem", "tudo bem", "salve", "olá!", "bem-vindo!", "seja bem-vindo!", "boa tarde!", "boa noite!", "oi, tudo bem?", "oi, como vai?", "oi, seja bem-vindo!", "olá, é um prazer te ver!", "oi, que bom que você veio!"],
        "responses": ["Olá, em que posso te ajudar?","Olá, como posso ajudar?","Hey, como posso ajudar?"],
        "context": [""]
      },
      {
        "tag": "welcome_problem",
        "patterns": ["pode me ajudar?","estou com problema","tenho um problema","me ajuda","estou com um problemão","tenho um problemão","estou com probleminha","tenho um probleminha","preciso de uma mão","me da uma mãozinha","preciso de uma ajudinha","problema","problemas","probleminha","problemão"],
        "responses": ["Pode descrever o seu problema?","Descreva para mim o seu problema, e farei o possível para ajudar.","Descreva o problema, para que eu possa entender melhor...","Para que eu possa entender melhor... poderia descrever qual seria o seu problema."],
        "context": [""]
      },
      {
        "tag": "who_are_you",
        "patterns": ["qual seu nome", "quem é você", "como você chama","o que você faz","o que é você","quem é vc","como vc se chama","o que vc faz"],
        "responses": ["Eu sou o chatbot de suporte da Semeq, e estou aqui para ajudar e resolver o seu problema!"],
        "context": [""]
      },
      {
        "tag": "thanks",
        "patterns": ["obrigada", "tks", "thank you", "valeu", "obrigada pela ajuda", "muito obrigada","obrigado", "ogd", "thank you","valeu","obrigado pela ajuda", "muito obrigado","obrigado!", "muito obrigado!", "agradeço muito!", "obrigada!", "muito obrigada!", "agradeço muito!", "obrigado por sua ajuda!", "muito obrigado por sua ajuda!", "agradeço muito por sua ajuda!", "obrigado pela resposta!", "muito obrigado pela resposta!", "agradeço muito pela resposta!", "obrigado pela sua atenção!", "muito obrigado pela sua atenção!", "agradeço muito pela sua atenção!", "muito obrigado pela sua ajuda, você foi muito útil!", "agradeço muito pela sua ajuda, você foi muito útil!","vlw"],
        "responses": ["De nada! Fico feliz em poder ajudar. Se tiver mais alguma dúvida, é só perguntar.", "Foi um prazer ajudar!", "Estou aqui para ajudar sempre que precisar.", "Sempre às ordens!", "Não precisa agradecer, é meu trabalho.", "O prazer foi todo meu.", "Qualquer coisa, estou aqui para ajudar."],
        "context": [""]
      },
      {
        "tag": "anything_else",
        "patterns": ["anything_else"],
        "responses": ["Desculpa, não entendi, tente novamente!","Desculpe, não consegui entender sua pergunta. Poderia reformulá-la de outra maneira?","Sinto muito, mas não consegui entender o que você quis dizer. Você pode reformular?","Desculpe-me, mas não entendi qual é a sua solicitação. Você poderia reformulá-la, por favor?","Desculpe, não entendi a sua pergunta. Você poderia fornecer mais informações ou esclarecer o que você gostaria de saber?"],
        "context": [""]
      },
      {
        "tag": "bye",
        "patterns": ["tchau", "bye", "falou", "tchau tchau", "bye bye", "até mais", "a gente se ve",  "até logo", "adeus", "exit", "sair", "sai", "flw", "xau", "encerrar", "finalizar", "parar", "cancelar", "desligar", "terminar"],
        "responses": ["Até a próxima!", "Até mais!", "Foi um prazer ajudá-lo(a)!", "Tenha um ótimo dia!", "Não hesite em me contatar novamente se precisar de ajuda!","Até mais! Se precisar de ajuda novamente, é só me chamar!"],
        "context": [""]
      }
    ]

add_json_to_db(json_data,"database\\chatbot.db")