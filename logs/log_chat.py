import os
import json

def log_chat(pattern, context, subcontext, response,first_question, subject, device, interface, model, problem, list_indice, indice):
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    if os.path.exists(json_path):
        if os.path.isfile(os.path.join(json_path, 'log.json')):
            with open(os.path.join(json_path, 'log.json'), 'r+', encoding='utf-8') as f:
                log = json.load(f)
                log.append({
                    "pattern": pattern,
                    "context": context,
                    "subcontext": subcontext,
                    "response": response,
                    "first_question": first_question,
                    "subject": subject,
                    "device": device,
                    "interface": interface,
                    "model": model,
                    "problem": problem,
                    "list_indice": list_indice,
                    "indice": indice
                })
                f.seek(0)
                json.dump(log, f, indent=4)
        else:
            with open(os.path.join(json_path, 'log.json'), 'w', encoding='utf-8') as f:
                log = [{
                    "pattern": pattern,
                    "context": context,
                    "subcontext": subcontext,
                    "response": response,
                    "first_question": first_question,
                    "subject": subject,
                    "device": device,
                    "interface": interface,
                    "model": model,
                    "problem": problem,
                    "list_indice": list_indice,
                    "indice": indice
                }]
                json.dump(log, f, indent=4)

def clear_log():
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    with open(os.path.join(json_path, 'log.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)