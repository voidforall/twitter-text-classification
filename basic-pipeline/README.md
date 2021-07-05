# EPFL Machine Learning Project 2: Basic Models

## Requirements

```
python3
nltk
tqdm # for training process visulization
sklearn # for svm, logistic and nn models
```

Please put the following data files into the `data` folder

- `test_data.txt`
- `train_neg.txt`
- `train_neg_full.txt`
- `train_pos.txt`
- `train_pos_full.txt`
  - Above five files are in this google drive [link](https://drive.google.com/drive/folders/1_1Sil23qLiVbCH-eRL5fNC917J_vUHll?usp=sharing). These files are preprocessed data file preprocessed by code in deep-pipeline. 
- `glove.twitter.27B.200d.txt`
- `glove.6B.50d.txt`
  - Above two could be downloaded from: https://nlp.stanford.edu/projects/glove/

## Directory Tree

```
│   build_voca_to_glove.py
│   data_utils.py
│   run.py
│   framework.py
│   README.md
│
├───data
│       glove.twitter.27B.200d.txt
│       glove.6B.50d.txt
│       sample_submission.csv
│       sub_vocab_cut.txt
│       test_data.txt
│       train_neg.txt
│       train_neg_full.txt
│       train_pos.txt
│       train_pos_full.txt
```

## File Description

- `build_voca_to_glove.py`
  - A Python version of combination of course offered data preprocessing code `build_vocab.sh`, `cooc.py`, `cut_vocab.sh`, `glove_template.py` and `pickle_vocab.py`.
- `data_utils.py`
  - Helper function related to data. 
- `framework.py`
  - Main entry to train prediction model. 
- `Data`
  - Data folder which contains the data files we need for training and testing.

## Features: Lemmatisation

Offer the option of `use_lemmatization` to use lemmatization for tokenization.

## HOW-TO USE

### To train your GloVE

- If you want to use the twitter subset to train the GloVE

```shell
python3 build_voca_to_glove.py --usesubset true
```

- If you want to use the complete dataset

```shell
python3 build_voca_to_glove.py --usesubset false
```

#### What to expect?

You could expect a data file `data/embeddings_sub.npy` or `embeddings_full.npy` is created. 

### To train the Logistic model

```shell
python run.py --model_name logistic --usesubset 0
```

### To train SVM

```shell
python run.py --model_name svm --usesubset 0 
```

### To train Neural Networks

The hidden layer of the neural network is set to 50, 10, 2, decreasing from 200 by 4 every time.   

```shell
python run.py --model_name nn --usesubset 0
```

### What to expect?

An accuracy and result shown.

### To use the pretrained GloVE

You can also use pretrained GloVE

```shell
python run.py --model_name svm --usesubset 0 --embeding twitter200
```

### To use the full dataset

You can do it by simply add a `--subset false` command line  parameter.

## HOW-TO Replicate

To replicate the result in our report. We are using Global Vectors for Word Representation (GloVE) pretrained on twitter dataset [[link](http://nlp.stanford.edu/data/glove.twitter.27B.zip)] to make it comparable with our benchmark deep models.

```shell
python run.py --model_name logistic --usesubset 0 --embeding twitter200
python run.py --model_name svm --usesubset 0 --embeding twitter200
python run.py --model_name nn --usesubset 0 --embeding twitter200
```

## References

- NumPy
  - Harris, C.R., Millman, K.J., van der Walt, S.J. et al. *Array programming with NumPy*. Nature 585, 357–362 (2020). DOI: [0.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2). ([Publisher link](https://www.nature.com/articles/s41586-020-2649-2)).
- Scikit-learn
  - F.   Pedregosa,   G.   Varoquaux,   A.   Gramfort,   V.   Michel,B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss,V.   Dubourg,   J.   Vanderplas,   A.   Passos,   D.   Cournapeau,M.  Brucher,  M.  Perrot,  and  E.  Duchesnay,  “Scikit-learn:Machine  learning  in  Python,”Journal  of  Machine  LearningResearch, vol. 12, pp. 2825–2830, 2011.
- GloVe
  - Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).
- NLTK
  - Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
- tqdm
  - da Costa-Luis, Casper O. (2019, September 17). tqdm: A fast, Extensible Progress Bar for Python and CLI (Version v4.36.0). Zenodo. http://doi.org/10.5281/zenodo.3435774

