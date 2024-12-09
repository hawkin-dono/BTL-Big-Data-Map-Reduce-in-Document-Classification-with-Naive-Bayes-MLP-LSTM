from mrjob.job import MRJob
from mrjob.step import MRStep
import re
import math
import util
from collections import defaultdict
import logging
import json

class NaiveBayesEvaluate(MRJob):
    FILES = ['util.py']
    def __init__(self, *args, **kwargs):
        super(NaiveBayesEvaluate, self).__init__(*args, **kwargs)
        logging.basicConfig(level=logging.DEBUG)

    def mapper_init(self):
        self.label_probs = {}
        self.word_label_probs = {}
        with open("/home/hd_user/storage/BTL-Big-Data-Map-Reduce-in-Document-Classification-with-Naive-Bayes-MLP-LSTM/output.json", 'r') as file:
            data = json.loads(file.read())
            # self.smooth_value = data['smooth_value']
            # self.vocab_size = data['vocab_size']
            self.label_probs = data['label_probs']
            self.word_label_probs = data['word_label_probs']
    def mapper(self, _, line):
        data = line.split(",")
        if len(data) < 3: return
        # label_text = data[-1]
        target_label = data[-2]
        text = ",".join(data[:-2])
        if text == "text": return
        else:
            text = util.extract(text)
            # self.label_count[label] += 1
            # for word in text:
            #     self.word_count[word] += 1
            #     self.word_label_count[word, label] += 1
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
                "text" : text,
                "logits" : label_probs
            }
    def reducer(self, key, values):
        count = 0
        correct = 0
        for value in values:
            if value['predict'] == value['target']:
                correct += 1
            count += 1
        yield None, {
            "correct" : correct,
            "count" : count,
            "accuracy" : correct / count
        }

if __name__ == '__main__':
    NaiveBayesEvaluate.run()
