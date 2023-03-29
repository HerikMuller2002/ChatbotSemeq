import os
import json

def log_chat(pattern,context,response,subject,device,interface,model,problem):
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    if os.path.exists(json_path):
        if os.path.isfile(os.path.join(json_path, 'log.json')):
            with open(os.path.join(json_path, 'log.json'), 'r+', encoding='utf-8') as f:
                log = json.load(f)
                log.append({
                    "pattern": pattern,
                    "context": context,
                    "response": response,
                    "subject": subject,
                    "device": device,
                    "interface": interface,
                    "model": model,
                    "problem": problem
                })
                f.seek(0)
                json.dump(log, f, indent=4)
        else:
            with open(os.path.join(json_path, 'log.json'), 'w', encoding='utf-8') as f:
                log = [{
                    "pattern": pattern,
                    "context": context,
                    "response": response,
                    "subject": subject,
                    "device": device,
                    "interface": interface,
                    "model": model,
                    "problem": problem
                }]
                json.dump(log, f, indent=4)

def clear_log():
    json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    with open(os.path.join(json_path, 'log.json'), 'w', encoding='utf-8') as f:
        json.dump([], f)