from pipeline import evaluate_debug, evaluate_hadoop, evaluate_local
import config as cf

# evaluate_local(mode='infer', path=cf.TEMP_INPUT_PATH)

evaluate_debug(mode='infer', path=cf.TEMP_INPUT_PATH)
