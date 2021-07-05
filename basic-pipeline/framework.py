# This is the main framework for the basic pipeline, here we do not rely on high level framework like allennlp

import os
import numpy as np

import sklearn

from data_utils import load_lines_from_file
from build_voca_to_glove import MyTokenization

# Modify this path to your datapath
DATA_BASE = "data/"
CROSSVALIDATION = True # If you would like to output prediction on test dataset, you can set it to False

# random seed fixed for replication
np.random.seed(2020)

# -------------------------------------------
# import data helper function
# -------------------------------------------
# load a numpy data file based on relative path and base path
# -------------------------------------------
def load_world_embedding(path):
    # support path input 
    path = os.path.join(DATA_BASE, path)
    word_embedding_data = np.load(path)
    return word_embedding_data

# -------------------------------------------
# load vocabulary
# -------------------------------------------
# load vocabulary and delete meaningless words
# -------------------------------------------
def load_vocabulary(path="vocab_cut.txt"):
    # read vocabulary from txt file 
    path = os.path.join(DATA_BASE, path)
    vocabulary = load_lines_from_file(path)
    # # delete '\n' words
    vocabulary = [i.strip("\n") for i in vocabulary]
    return vocabulary

# -------------------------------------------
# load_tweets
# -------------------------------------------
# load_tweets according to subset or complete data
# -------------------------------------------
def load_tweets(use_subset=True):
    if use_subset == True:
        train_neg = load_lines_from_file(os.path.join(DATA_BASE, "train_neg.txt"))
        train_pos = load_lines_from_file(os.path.join(DATA_BASE, "train_pos.txt"))
        test_data  = load_lines_from_file(os.path.join(DATA_BASE, "test_data.txt"))
    else:
        train_neg = load_lines_from_file(os.path.join(DATA_BASE, "train_neg_full.txt"))
        train_pos = load_lines_from_file(os.path.join(DATA_BASE, "train_pos_full.txt"))
        test_data  = load_lines_from_file(os.path.join(DATA_BASE, "test_data.txt"))

    return train_neg, train_pos, test_data

# -------------------------------------------
# sentences feature construction
# -------------------------------------------
# construct features from sentence to learn
# -------------------------------------------
def sentences_to_features(sentences, dict_vocabulary_to_embedding, myTokenization, DIM_FEATURE):
    # return feature or that sentence according to world_vector_data
    # now we support averaging word's vector to get the sentence vector (as required in the project README.md)
    assert DIM_FEATURE > 0
    
    dataset_features = np.zeros([len(sentences), DIM_FEATURE])
    for sentence_index, sentence in enumerate(sentences):
        words_in_sentece = myTokenization.process(sentence) # sentence.strip().split(" ")
        counter = 0

        for word in words_in_sentece:
            if word in dict_vocabulary_to_embedding:
                dataset_features[sentence_index] += dict_vocabulary_to_embedding[word]
                counter += 1
        
        if counter != 0:
            dataset_features[sentence_index] /= counter
    # Test
    counter = 0
    test_features = np.zeros([DIM_FEATURE])
    for word in myTokenization.process(sentences[1]): 
        if word in dict_vocabulary_to_embedding:
            test_features += dict_vocabulary_to_embedding[word]
            counter += 1
    assert np.isclose(test_features/counter, dataset_features[1]).all(), "feature calculation error: not as expected"
    
    return dataset_features

# -------------------------------------------
# load pretrained GloVE
# -------------------------------------------
# use pretrained GloVe embedding
# -------------------------------------------
def read_GloVe_data(file_name="data/glove.6B.50d.txt"):
    with open(file_name,'r', encoding="utf8") as f:
        word_vocab = set() # not using list to avoid duplicate entry
        word2vector = {}
        for line in f:
            line_ = line.strip() #Remove white space
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
    print("Total Words in DataSet:",len(word_vocab))
    return word_vocab,word2vector

