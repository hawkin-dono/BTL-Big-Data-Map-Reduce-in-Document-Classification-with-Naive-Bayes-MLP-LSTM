from bayes import NaiveBayesTrainer
from bayes_eval import NaiveBayesEvaluate
import logging
import os
import time
import config as cf
from typing import Literal

def run_debug():
    __run([])
def run_local():
    extra_args = ["-cmrjob.conf", "-rlocal"]
    __run(extra_args)
def run_hadoop():
    extra_args = ["-cmrjob.conf", "-rhadoop"]
    __run(extra_args)
def evaluate_debug(mode: Literal['eval', 'infer'], path: str = None):
    __evaluate([], mode, path)
def evaluate_local(mode: Literal['eval', 'infer'], path: str = None):
    extra_args = ["-cmrjob.conf", "-rlocal"]
    __evaluate(extra_args, mode, path)
def evaluate_hadoop(mode: Literal['eval', 'infer'], path: str = None):
    extra_args = ["-cmrjob.conf", "-rhadoop"]
    __evaluate(extra_args, mode, path)
def __evaluate(extrargs, mode: Literal['eval', 'infer'], path: str):
    args = []
    if path is None:
        args.append(cf.RAW_TEST_DATA_PATH)
    else:
        args.append(path)
    args.extend(extrargs)
    if mode == 'infer':
        args.extend(["--is_infer", "1"])
    else:
        args.extend(["--is_infer", "0"])
    job = NaiveBayesEvaluate(args=args)
    with job.make_runner() as runner:
        runner.run()
def __run(extra_args):
    logging.getLogger('mrjob').setLevel(logging.WARNING)
    def run_new():
        import shutil
        folder_paths = []
        if not os.path.exists(cf.ABS_OUTPUT_PATH): os.mkdir(cf.ABS_OUTPUT_PATH)
        for folder_path in folder_paths:
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
            os.makedirs(folder_path)
        file_paths = [
            os.path.join(cf.ABS_OUTPUT_PATH, f"model_weight.{cf.FILE_EXTENSION}"),
            os.path.join(cf.ABS_OUTPUT_PATH, f"metadata.json"),
            os.path.join(cf.ABS_OUTPUT_PATH, "evaluate.json")
        ]
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
    run_new()
    args = [cf.RAW_TRAIN_DATA_PATH]
    args.extend(extra_args)
    start_time = time.time()
    job = NaiveBayesTrainer(args=args)
    with job.make_runner() as runner:
        runner.run()
    print(f"Train time {time.time()-start_time} s")
    args = [cf.RAW_TEST_DATA_PATH]
    args.extend(extra_args)
    args.extend(["--is_infer", "0"])
    job = NaiveBayesEvaluate(args=args)
    with job.make_runner() as runner:
        runner.run()