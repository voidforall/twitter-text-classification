import numpy as np 
import sys
from TweetReader import TweetReader 
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataloader import PyTorchDataLoader

"""
This file is used for test the TweetReader.py for ../train.py.
"""

np.set_printoptions(threshold=sys.maxsize)

reader = TweetReader(segment_sentences=False, max_sequence_length=64)
print("Loading dataset...")
file_path = r"../data/test_subset.json"

dataset = reader.read(file_path)

print("Building vocabulary from the dataset...")
vocab = Vocabulary.from_instances(instances=dataset, min_count={"tokens":10})
print("Temporary vocabulary has been built.")

# save the vocabulary__init__
vocab.save_to_files(r"../data/vocab/")

print(dataset[0])
