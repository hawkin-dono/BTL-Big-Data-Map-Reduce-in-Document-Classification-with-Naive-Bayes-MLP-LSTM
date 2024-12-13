from mlp.set_cwd import set_cwd
import subprocess
import os
import mlp.config as cf
import json
from typing import Literal
import time


set_cwd()
cwd = os.getcwd()

def inter_text(text: str):
    
    data = text + ",0,empty"
    path = cf.TEMP_INPUT_PATH
    with open(path, 'w') as file: file.write(data)

    cmds = ["cd mlp","python3 infer_temp.py","cd .."]
    result = subprocess.run(" && ".join(cmds), shell=True, cwd=cwd, capture_output=True, text=True)
    path = os.path.join(cf.ABS_OUTPUT_PATH, "infer_result.json")
    with open(path, 'r') as file:
        result = json.loads(file.read())
    return result

text = "Bobcats Trade Drobnjak to Hawks for Pick (AP) AP - The Charlotte Bobcats traded center Predrag Drobnjak to the Atlanta Hawks on Monday for a second round pick in the 2005 NBA draft.,"
start_time = time.time()
result = inter_text(text)
single_time = time.time() - start_time
print(result['detail'])