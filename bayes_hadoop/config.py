from typing import Literal
import os

PROJECT_PATH = None

if os.path.exists('cwd.txt'):
    with open('cwd.txt', 'r') as file:
        PROJECT_PATH = file.read()
else:
    with open('bayes_hadoop/cwd.txt', 'r') as file:
        PROJECT_PATH = file.read()

ABS_OUTPUT_PATH = os.path.join(PROJECT_PATH, "output/bayes_hadoop")
RAW_TRAIN_DATA_PATH = os.path.join(PROJECT_PATH, "AllData/ag_news_data/train.csv")
RAW_TEST_DATA_PATH = os.path.join(PROJECT_PATH, "AllData/ag_news_data/test.csv")
TEMP_INPUT_PATH = os.path.join(ABS_OUTPUT_PATH, "temp.txt")
SMOOTHING = 1

CLUSTERS = 4
MODE : Literal["pickle", "json"] = "json"
FILE_EXTENSION = "pkl" if MODE == "pickle" else "json"
READMODE = lambda x: "rb" if x.endswith("pkl") else "r"
WRITEMODE = lambda x: "wb" if x.endswith("pkl") else "w"

FORCE_JSON_INPUT = True