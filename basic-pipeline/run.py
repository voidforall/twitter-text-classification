import os
import numpy as np

import sklearn

from data_utils import load_lines_from_file
from build_voca_to_glove import MyTokenization

from framework import *

# -------------------------------------------
# main entry for Linear Classifier
# -------------------------------------------
if __name__ == "__main__":
    # -------------------------------------------
    # argument parser
    # -------------------------------------------
    # commandline parser to choose using pretrain or not
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

    parser.add_argument("--embeding", type=str, nargs='?',
                            default="subset",
                            help="Choose one from [subset, full, wiki50, twitter200], subset means embeding by subset")
    parser.add_argument("--usesubset", type=str2bool, nargs='?',
                            default=True,
                            help="Use 1 or y if you want to use subset for training data. True by default")
    parser.add_argument("--model_name", type=str, nargs='?',
                            default="nn",
                            help="Choose one from [svm, logistic, nn]")
    args = parser.parse_args()

    print("[embeding way]:", args.embeding)
    print("[using model]:", args.model_name)

    from time import time
    time_start_program = time()
    
    # load word vector
    if args.embeding == "wiki50":
        vocab, w2v = read_GloVe_data()
        dict_vocabulary_to_embedding = w2v
        myTokenization = MyTokenization(use_lemmatization=False, use_stem=True, use_TweetTokenizer=True) # do tokenization, default param: use_lemmatization=True, use_stem=False, use_TweetTokenizer=True
        DIM_FEATURE = len(w2v[next(iter(vocab))])
    elif args.embeding == "twitter200":
        vocab, w2v = read_GloVe_data("data/glove.twitter.27B.200d.txt")
        dict_vocabulary_to_embedding = w2v
        myTokenization = MyTokenization(use_lemmatization=False, use_stem=True, use_TweetTokenizer=True) # do tokenization, default param: use_lemmatization=True, use_stem=False, use_TweetTokenizer=True
        DIM_FEATURE = len(w2v[next(iter(vocab))])
    else:
        if args.embeding == "subset":
            embeding_path = "embeddings_sub.npy"
        elif args.embeding == "full":
            embeding_path = "embeddings_full.npy"
        else:
            raise NotImplementedError("invalid embeding type")

        word_embedding_data = load_world_embedding(embeding_path)
        
        print("[word_embedding_data loaded] shape:", word_embedding_data.shape)
        vocabulary = load_vocabulary()
        print("[vocabulary loaded] length:", len(vocabulary))
        # alignment
        dict_vocabulary_to_embedding = dict(zip(vocabulary, word_embedding_data))
        myTokenization = MyTokenization(use_lemmatization=True, use_stem=False, use_TweetTokenizer=True) # do tokenization, default param: use_lemmatization=True, use_stem=False, use_TweetTokenizer=True
        # DIM_FEATURE = len(dict_vocabulary_to_embedding[0])
        DIM_FEATURE = word_embedding_data.shape[1]

    # load in data
    train_neg, train_pos, test_data = load_tweets(args.usesubset)
    print("[train_neg loaded] length:", len(train_neg))
    print("[train_pos loaded] length:", len(train_pos))
    print("[test_data loaded] length:", len(test_data))

    train_neg_features = sentences_to_features(train_neg, dict_vocabulary_to_embedding, myTokenization, DIM_FEATURE=DIM_FEATURE)
    train_pos_features = sentences_to_features(train_pos, dict_vocabulary_to_embedding, myTokenization, DIM_FEATURE=DIM_FEATURE)
    test_data_features = sentences_to_features(test_data, dict_vocabulary_to_embedding, myTokenization, DIM_FEATURE=DIM_FEATURE)
    print("[Sentences to features completed]")

    # construct dataset
    train_data = np.concatenate([np.array(train_pos_features), np.array(train_neg_features)], axis=0) 
    # train_data = np.stack([np.array(train_pos_features), np.array(train_neg_features)], axis=0) 
    
    train_data = train_data.reshape(-1, DIM_FEATURE)
    train_label = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg))) # svm的输入是0，1

    # shuffle dataset
    train_data_shuffle, train_label_shuffle = sklearn.utils.shuffle(train_data, train_label, random_state=2020)
    print("[Shuffle data completed]")
    print("[Begin training]")
    
    if CROSSVALIDATION == False:
        # -------------------------------------------
        # do simple training and testing and write the test prediction to file
        # -------------------------------------------
        if args.model_name == "svm":
            from sklearn import svm
            clf = svm.SVC(kernel='rbf') 
            print("[using model]: svm(rbf)")
            clf.fit(train_data_shuffle, train_label_shuffle)
            prediction = clf.predict(test_data_features)

        elif args.model_name == "logistic":
            from sklearn import linear_model
            print("[using model]: logistic")
            logistic = linear_model.LogisticRegression()
            logistic.fit(train_data_shuffle, train_label_shuffle)
            prediction = logistic.predict(test_data_features)

        elif args.model_name == "nn":
            print("[using model]: neural network")
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 10, 2), random_state=1, max_iter=1)
            clf.fit(train_data_shuffle, train_label_shuffle)
            prediction = clf.predict(test_data_features)
        else:
            raise NotImplementedError("invalid model name:", args.model_name)
    
        #write to file
        from result_helpers import create_csv_submission
        # create_csv_submission 
        labels = [label if label==1 else -1 for label in prediction]
        ids = list(range(1, len(labels) + 1))

        output_file_path = os.path.join('data', "submission.csv")
        create_csv_submission(ids, labels, output_file_path)

    else:
        # -------------------------------------------
        # do cross-validation
        # -------------------------------------------
        from sklearn.model_selection import cross_val_score
        
        if args.model_name == "svm":
            from sklearn import svm
            clf = svm.SVC(kernel='rbf') 
            print("[using model]: svm(rbf)")
            scores_f1 = cross_val_score(clf, train_data_shuffle, train_label_shuffle, scoring="f1", cv=5, n_jobs=-1, verbose=True)
            scores_accuracy = cross_val_score(clf, train_data_shuffle, train_label_shuffle, scoring="accuracy", cv=5, n_jobs=-1, verbose=True)
        elif args.model_name == "logistic":
            from sklearn import linear_model
            print("[using model]: logistic")
            logistic = linear_model.LogisticRegression(max_iter=300)
            scores_f1 = cross_val_score(logistic, train_data_shuffle, train_label_shuffle, scoring="f1", cv=5, n_jobs=-1)
            scores_accuracy = cross_val_score(logistic, train_data_shuffle, train_label_shuffle, scoring="accuracy", cv=5, n_jobs=-1)
        elif args.model_name == "nn":
            print("[using model]: neural network")
            from sklearn.neural_network import MLPClassifier
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 10, 2), random_state=1, max_iter=10)
            scores_f1 = cross_val_score(clf, train_data_shuffle, train_label_shuffle, scoring="f1", cv=5, n_jobs=-1, verbose=True)
            scores_accuracy = cross_val_score(clf, train_data_shuffle, train_label_shuffle, scoring="accuracy", cv=5, n_jobs=-1, verbose=True)
        else:
            raise NotImplementedError("invalid model name:", args.model_name)

        print("[cross-validation F1 score:]", scores_f1)
        print("[cross-validation F1 score avg:]", scores_f1.mean())
        print("[cross-validation Accuracy score:]", scores_accuracy)
        print("[cross-validation Accuracy score avg:]", scores_accuracy.mean())

    time_end_program = time()

    print("[End of framework.py]")
    print("[Time consumed]:", time_end_program-time_start_program)