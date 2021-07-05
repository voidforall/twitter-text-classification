## This file is python version of shell script offered from the course lab materials
## We combine all the commandlines into this file

import os

from tqdm import tqdm
from data_utils import load_lines_from_file
import collections

# import nltk related
import nltk
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer 
# download nltk related files
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english')) 
lemmatizer = WordNetLemmatizer() 
ps = PorterStemmer()


# -------------------------------------------
# MyTokenization
# -------------------------------------------
# Tokenization to split sentence into words. We offer different ways of tokenization
# - use tweetTokenizer or not
# - use lemmatization or not
# - use stem or not
# -------------------------------------------
class MyTokenization(object):
    def __init__(self, use_lemmatization=True, use_stem=False, use_TweetTokenizer=True):
        self.use_lemmatization = use_lemmatization
        self.use_stem = use_stem
        self.use_TweetTokenizer = use_TweetTokenizer

        if use_TweetTokenizer == True:
            tknzr = TweetTokenizer() # use tweet tokenizer, <user> will be one token, instead of <, >, user. 
            self.tokenizer = tknzr.tokenize
        else:
            self.tokenizer = word_tokenize

        if use_lemmatization == True: # I think lemmatization is stronger than stem
            # https://www.geeksforgeeks.org/python-lemmatization-with-nltk/?ref=lbp
            self.token_func = lambda sentence: [lemmatizer.lemmatize(i) for i in self.tokenizer(sentence)] # https://www.geeksforgeeks.org/python-lemmatization-with-nltk/?ref=lbp
        elif use_stem == True: # https://www.geeksforgeeks.org/python-stemming-words-with-nltk/?ref=rp
            self.token_func = lambda sentence: [ps.stem(i) for i in self.tokenizer(sentence)] # https://www.geeksforgeeks.org/python-stemming-words-with-nltk/?ref=rp
        else:
            self.token_func = lambda sentence: self.tokenizer(sentence)

    def process(self, line):
        return self.token_func(line)
        
# commandline parser to choose sub or full set of tweeter
if __name__ == "__main__":
    # -------------------------------------------
    # argument parsing
    # -------------------------------------------
    # define a str2bool function for flexibility of command line parameters
    # users can chose to use subset to train GloVE or the complete dataset
    # -------------------------------------------
    import argparse
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser(description="parser")
    parser.add_argument("--subset", type=str2bool, nargs='?',
                            const=True, default=True,
                            help="Use 1 or y if you want to use subset. True by default")

    args = parser.parse_args()
    using_subset = args.subset
    if using_subset == True:
        TRAIN_POS_FILE = 'data/train_pos.txt'
        TRAIN_NEG_FILE = 'data/train_neg.txt'
    else:
        TRAIN_POS_FILE = 'data/train_pos_full.txt'
        TRAIN_NEG_FILE = 'data/train_neg_full.txt'


    # -------------------------------------------
    # to build a vocabulary
    # -------------------------------------------
    # From the commands offered in the course material
    # bash build_vocab.sh
    # cat ../data/twitter-datasets/train_pos.txt ../data/twitter-datasets/train_neg.txt | sed 's/ /\'$'\n/g' | grep -v "^\s*$" | sort | uniq -c > vocab.txt
    # -------------------------------------------
    def build_vocab(path_list, myTokenization=None):

        print("[Begin build vocabulary")
        vocabulary = []
        # vocabulary_tmp = []
        for path in path_list:
            for sentence in tqdm(load_lines_from_file(path)):
                vocabulary += myTokenization.process(sentence)

        return vocabulary

    # -------------------------------------------
    # counter
    # -------------------------------------------
    # bash cut_vocab.sh
    # cat vocab.txt | sed "s/^\s\+//g" | sed 's/^[ \t]*//g' |  sort -rn | grep -v "^[1234]\s" | cut -d' ' -f2 > vocab_cut.txt
    # -------------------------------------------
    def vocab_count(vocabulary, threshold=5):
        vocabulary_count = collections.Counter(vocabulary)
        vocabulary_count_output = {}
        
        for word in vocabulary_count.keys():
            if vocabulary_count[word] >= threshold and word not in stop_words:
                vocabulary_count_output[word] = vocabulary_count[word]

        sorted_vocabulary = sorted(vocabulary_count_output.items(), key=lambda x: x[1])
        sorted_vocabulary = sorted_vocabulary[::-1]

        return sorted_vocabulary

    def write_vocabulary(sorted_vocabulary):
        with open("data/vocab_cut.txt", "w") as f:
            for pair in sorted_vocabulary:
                f.write(pair[0] + os.linesep)
        
    # python3 pickle_vocab.py
    import pickle
    def pickle_vocab():
        vocab = dict()
        with open('data/vocab_cut.txt') as f:
            for idx, line in enumerate(f):
                vocab[line.strip()] = idx

        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

    # -------------------------------------------
    # compute the co-occurance matrix
    # -------------------------------------------
    def cooc(myTokenization):
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        vocab_size = len(vocab)

        data, row, col = [], [], []
        counter = 1
        
        for fn in [TRAIN_POS_FILE, TRAIN_NEG_FILE]:
            with open(fn, encoding="utf8") as f:
                for line in tqdm(f):
                    tokens = [vocab.get(t, -1) for t in myTokenization.process(line)]

                    tokens = [t for t in tokens if t >= 0]
                    for t in tokens:
                        for t2 in tokens:
                            data.append(1)
                            row.append(t)
                            col.append(t2)

                    if counter % 10000 == 0:
                        print(counter)
                    counter += 1

        cooc = coo_matrix((data, (row, col)))
        print("summing duplicates (this can take a while)")
        cooc.sum_duplicates()

        with open('data/cooc.pkl', 'wb') as f:
            pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)

    # python3 glove_solution.py
    from scipy.sparse import *
    import numpy as np
    import pickle

    # -------------------------------------------
    # glove solution
    # -------------------------------------------
    # use sgd to compute the glove solution
    # -------------------------------------------
    def glove_solution(using_subset):
        print("loading cooccurrence matrix")
        with open('data/cooc.pkl', 'rb') as f:
            cooc = pickle.load(f)
        print("{} nonzero entries".format(cooc.nnz))

        nmax = 100
        print("using nmax =", nmax, ", cooc.max() =", cooc.max())

        print("initializing embeddings")
        embedding_dim = 20
        # random initialization
        xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
        ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

        eta = 0.001
        alpha = 3 / 4

        if using_subset:
            epochs = 10
        else:
            epochs = 1

        # using SGD to find the glove solution
        for epoch in tqdm(range(epochs)):
            for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
                fn = min(1.0, (n / nmax) ** alpha)
                logn = np.log(n)
                x, y = xs[ix, :], ys[jy, :]
                xs[ix, :] = xs[ix, :] + 2 * eta * fn * (logn - np.dot(x, y)) * y
                ys[jy, :] = ys[jy, :] + 2 * fn * eta * (logn - np.dot(x, y)) * x
        
        if using_subset == True:
            np.save('data/embeddings_sub', xs)
        else:
            np.save('data/embeddings_full', xs)

# main entry
if __name__ == "__main__":
    if using_subset == True:
        print("[Warning]: Now using subset dataset")
    else:
        print("[Warning]: Now using total dataset")

    myTokenization = MyTokenization()

    vocabulary = build_vocab([TRAIN_POS_FILE, TRAIN_NEG_FILE], myTokenization)
    sorted_vocabulary = vocab_count(vocabulary)

    write_vocabulary(sorted_vocabulary)
    pickle_vocab()
    cooc(myTokenization)
    glove_solution(using_subset)

    print("[Program finished]")