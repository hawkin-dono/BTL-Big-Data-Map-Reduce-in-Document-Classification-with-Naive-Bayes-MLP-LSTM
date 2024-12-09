from mrjob.job import MRJob
import math
import util
from collections import defaultdict
import logging
import json
import config as cf
import os

class NaiveBayesEvaluate(MRJob):
    FILES = ['util.py', 'config.py', 'cwd.txt']
    def configure_args(self):
        super(NaiveBayesEvaluate, self).configure_args()
        self.add_passthru_arg("--is_infer", type=int, help="Is interence or evaluate")

    def __init__(self, *args, **kwargs):
        super(NaiveBayesEvaluate, self).__init__(*args, **kwargs)
        logging.basicConfig(level=logging.DEBUG)

    def mapper_init(self):
        self.label_probs = {}
        self.word_label_probs = {}
        file_path = os.path.join(cf.ABS_OUTPUT_PATH, f"model_weight.{cf.FILE_EXTENSION}")
        model_weight = util.reader(file_path)
        self.label_probs : dict[str, float] = model_weight['label_probs']
        self.word_label_probs : dict[str, dict[str, float]] = model_weight['word_label_probs']
    def mapper(self, _, line):
        data = line.split(",")
        if len(data) < 3: return
        label_text = data[-1]
        target_label = data[-2]
        target_label = label_text
        text = ",".join(data[:-2])
        if text == "text": return
        else:
            text = util.extract(text)
            label_probs = defaultdict(float)
            for word in text:
                word_label_prob = self.word_label_probs.get(word, self.word_label_probs['<unk>'])
                for label in self.label_probs:
                    label_probs[label] += math.log(word_label_prob[label] * self.label_probs[label])
            max_label = None
            max_prob = float('-inf')
            for label, prob in label_probs.items():
                if max_prob < prob:
                    max_prob = prob
                    max_label = label
            max_log_prob = max(label_probs.values())
            log_sum_exp = max_log_prob + math.log(
                sum(math.exp(prob - max_log_prob) for prob in label_probs.values())
            )
            yield _, {
                "predict": max_label,
                "target": target_label,
                "prob" : math.exp(max_prob - log_sum_exp),
                # "text" : text,
                "logits" : label_probs
            }
    def reducer_init(self):
        if self.options.is_infer == 1:
            self.mode = "infer"
        elif self.options.is_infer == 0:
            self.mode = "eval"
    def reducer(self, _, values):
        if self.mode == "infer":
            self.results = []
        count = 0
        correct = 0
        for value in values:
            if value['predict'] == value['target']:
                correct += 1
            count += 1
            if self.mode == "infer":
                self.results.append(value)
        self.evaluate_log = {
            "correct" : correct,
            "count" : count,
            "accuracy" : correct / count
        }
    def reducer_final(self):
        file_path = os.path.join(cf.ABS_OUTPUT_PATH, f"model_weight.{cf.FILE_EXTENSION}")
        model_weight = util.reader(file_path)
        metadata : dict[str, object] = model_weight
        metadata.pop('label_probs')
        metadata.pop('word_label_probs')
        self.evaluate_log["metadata"] = metadata
        if self.mode == "infer":
            file_path = os.path.join(cf.ABS_OUTPUT_PATH, f"infer_result.json")
            self.results.insert(0, self.evaluate_log)
            util.writer(self.results, file_path)
        else:
            file_path = os.path.join(cf.ABS_OUTPUT_PATH, f"evaluate.json")
            util.writer(self.evaluate_log, file_path)
if __name__ == '__main__':
    NaiveBayesEvaluate.run()
