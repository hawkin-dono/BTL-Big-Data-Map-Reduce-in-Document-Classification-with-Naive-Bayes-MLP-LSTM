from typing import Literal
import os

PROJECT_PATH = "D:\\Programming\\Storage\\Class\\2024-2025\\BigData\\BTL-Big-Data-Map-Reduce-in-Document-Classification-with-Naive-Bayes-MLP-LSTM"

ABS_OUTPUT_PATH = os.path.join(PROJECT_PATH, "output/mlp")
RAW_TRAIN_DATA_PATH = os.path.join(PROJECT_PATH, "ag_news_data/train.csv")
RAW_TEST_DATA_PATH = os.path.join(PROJECT_PATH, "ag_news_data/test.csv")
TRAIN_DATA_PATH = os.path.join(ABS_OUTPUT_PATH, "train")
TEST_DATA_PATH = os.path.join(ABS_OUTPUT_PATH, "test")
VOCAB_PATH = os.path.join(ABS_OUTPUT_PATH, "vocab.json")

EMBED_DIM = 32
SEQ_LENGTH = 32
HIDDEN_SIZE = 256
OUTPUT_SIZE = 4
LR_MULTIPLER = 10 
EMBED_LR = 1e-3 * LR_MULTIPLER
FC1_LR = 1e-3 * LR_MULTIPLER
FC2_LR = 1e-3 * LR_MULTIPLER

CLUSTERS = 4
MODE : Literal["pickle", "json"] = "pickle"
FILE_EXTENSION = "pkl" if MODE == "pickle" else "json"
READMODE = lambda x: "rb" if x.endswith("pkl") else "r"
WRITEMODE = lambda x: "wb" if x.endswith("pkl") else "w"

FORCE_JSON_INPUT = True