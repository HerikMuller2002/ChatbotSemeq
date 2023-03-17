import pandas as pd
import os

csv_censored_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ChatbotSemeq\\database\\tabela_problemas_csv.csv"))
dados = pd.read_csv("database\\tabela_problemas_csv.csv", encoding="iso-8859-1")
print(dados)