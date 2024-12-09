import re, json, logging, pickle
from config import *
def printf(text):
    logging.debug(text)
def extract(text: str) -> list[str]:
    return re.sub(r'[^a-z\s]', '', text.lower()).split()
def writer(data: object, file_path: str):
    write_mode = 'wb' if file_path.endswith('pkl') else 'w'
    if write_mode == 'w':
        try:
            with open(file_path, write_mode) as file:
                file.write(json.dumps(data))
        except Exception as ex:
            print("Path" + file_path)
            print(ex)

    else:
        with open(file_path, write_mode) as file:
            pickle.dump(data, file)
def reader(file_path: str):
    read_mode = 'rb' if file_path.endswith('pkl') else 'r'
    if read_mode == 'r':
        with open(file_path, read_mode) as file:
            data = json.loads(file.read())
    else:
        with open(file_path, read_mode) as file:
            data = pickle.load(file)
    return data