from mrjob.job import MRJob
from mrjob.step import MRStep
import re
import math
import util
from collections import defaultdict
import logging

class NaiveBayesTrainer(MRJob):
    FILES = ['util.py']
    def __init__(self, *args, **kwargs):
        super(NaiveBayesTrainer, self).__init__(*args, **kwargs)
        logging.basicConfig(level=logging.DEBUG)

    # Formula :
    # P(C|X) = P(X|C) * P(C) / P(X)
    # Simple
    # P(X) = count(word) / total_word
    # P(C) = count(label) / total_label
    # P(X|C) = count(word, label) / each_label_count
    # Laplace
    # P(X) = count(word) / total_word 
    # P(C) = (count(label) + 1) / ( total_label + label_size)
    # P(X|C) = (count(word, label) + 1) / (each_label_count + vocab_size)
    def mapper_word_count(self, _, line):
        data = line.split(",")
        if len(data) < 3: return
        # label_text = data[-1]
        label = data[-2]
        text = ",".join(data[:-2])
        if text == "text": return
        else:
            text = util.extract(text)
            yield f'${label}$', 1
            for word in text:
                yield word, 1
                yield (word, label), 1
    def reducer_word_count(self, key, values):
        yield key, sum(values)
    def mapper_naive_bayes(self, key, values):
        yield None, (key, values)
    def reducer_naive_bayes(self, _, values):
        word_count = defaultdict(int)
        label_count = defaultdict(int)
        word_label_count = defaultdict(lambda: defaultdict(int))
        for key, value in values:
            if not isinstance(key, str):
                word_label_count[key[0]][key[1]] = value
            elif key[0] == '$' and key[-1] == '$':
                label_count[key[1:-1]] = value
            else:
                word_count[key] = value
        smooth_value = 1
        vocab_size = len(word_count) + smooth_value
        
        label_probs = {}
        total_labels = sum(label_count.values())
        for key, count in label_count.items():
            label_probs[key] = (count + smooth_value) / ( total_labels + len(label_count))

        word_probs = {}
        total_words = sum(word_count.values())
        for key, count in word_count.items():
            word_probs[key] = count / total_words

        word_label_probs = {}
        for word, item in word_label_count.items():
            _word_label_probs = {}
            for label in label_count.keys():
                count = item.get(label, 0)
                _word_label_probs[label] = (count + smooth_value )/ (label_count[label] + smooth_value * vocab_size)
            word_label_probs[word] = _word_label_probs
        word_label_probs['<unk>'] = {label : smooth_value / (label_count[label] + vocab_size * smooth_value) for label in label_count}
        self.label_probs = label_probs
        self.word_label_probs = word_label_probs
        yield None, {
            "smooth_value" : smooth_value,
            "vocab_size" : vocab_size,
            "label_probs" : label_probs,
            "word_label_probs" : word_label_probs
        }
    def steps(self):
        return [
            MRStep(mapper=self.mapper_word_count, reducer=self.reducer_word_count),
            MRStep(mapper=self.mapper_naive_bayes, reducer=self.reducer_naive_bayes)
        ]
if __name__ == '__main__':
    NaiveBayesTrainer.run()
