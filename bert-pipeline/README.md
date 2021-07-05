# Project 2 Text Classification

> Code has been tested on MacOS, Linux Ubuntu and Windows 10 platform

### Requirements

```
python=3.6.5
torch==1.7.0
transformers==4.0.0
tokenizers
sklearn
```

### Directory Tree

```
│   README.md
│   extract.csv
│   preprocess.py
│   dataprepare.py
│   bert_extract_train.py
│   bert_extract_infer.py
│   bert_train.py
│   bert_evaluate.py
│   roberta_train.py
│   roberta_evaluate.py
│   bert_train_exaug.py
│   bert_evaluate_exaug.py
│   bert_train_exsiamense.py
│   bert_evaluate_exsiamense.py
│   bert_evaluate_exsiamense.py
│   bert_ensemble.py
│
├── saved
|       ...
|       
|

```

## File Description

- `extract.csv`：The Tweet Sentiment Extraction dataset from [Kaggle Competition](https://www.kaggle.com/c/tweet-sentiment-extraction/data).
- `preprocess.py`：Preprocess script for parsing tweet data.
- `dataprepare.py`： BERT tokenization and other data preprocessing script.
- `bert_extract_train.py`：BERT-based sentiment extraction training script.
- `bert_extract_infer.py`：BERT-based sentiment extraction inference script.
- `bert_train.py/bert_evaluate.py`：BERT-based single-layer sentiment classifier training/evaluation script.
- `roberta_train.py/roberta_evaluate.py`：RoBERTa-based multi-layer sentiment classifier training/evaluation script.
- `bert_train_exaug.py/bert_evaluate_exaug.py`：BERT-based extraction-augmented sentiment classifier (pair sequences strategy) training/evaluation script.
- `bert_train_exsiamense.py/bert_evaluate_exsiamense.py`：BERT-based extraction-augmented sentiment classifier (siamense strategy) training/evaluation script.

## HOW-TO USE

### To prepare the data

```shell
:~/bert-pipeline$ cp -r ../deep-pipeline/data/ ./data/
:~/bert-pipeline$ python bert_extract_train.py
:~/bert-pipeline$ python bert_extract_infer.py
```
### What to expect?

First, copy the preprocessed data in the `deep-pipeline`. The data is enough for training the BERT-based models without sentiment extraction augmentation.

To train the sentimen extraction augmented models, it is required to train the sentiment extractor first with `bert_extract_train.py`. Then, running inference script `bert_extract_infer.py` to augment the original data. The dataset will be saved in `./data/` folder.

### To train the BERT-based models

```shell
:~/bert-pipeline$ python bert_train.py
:~/bert-pipeline$ python roberta_train.py
:~/bert-pipeline$ python bert_train_exaug.py
:~/bert-pipeline$ python bert_train_exsiamense.py
```
### What to expect?

The scripts above will be used to fine-tune the BERT-based models on pre-trained weights. Trained models will be saved in `./saved/model_name/` folder, and the performance on training set/validation set will be reported.

### To evaluate the BERT-based models

```shell
:~/bert-pipeline$ python bert_evaluate.py
:~/bert-pipeline$ python roberta_evaluate.py
:~/bert-pipeline$ python bert_evaluate_exaug.py
:~/bert-pipeline$ python bert_evaluate_exsiamense.py
```
### What to expect?

The scripts above will be used to evaluate the BERT-based models on fine-tuned weights. The performance on test set will be reported, and the script also generates `submission.csv` under the saved folder.

### To ensemble the trained BERT-based models

```shell
:~/bert-pipeline$ python bert_ensemble.py
```

### What to expect?

To use the model ensembling, it is required to train the above models, and evaluate their results. `bert_ensemble.py` will use bagging strategy to ensemble their results to generate the final ensembled `submission.csv` file, also with the reported performance on the test set.

## References
- NumPy
  - Harris, C.R., Millman, K.J., van der Walt, S.J. et al. *Array programming with NumPy*. Nature 585, 357–362 (2020). DOI: [0.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2). ([Publisher link](https://www.nature.com/articles/s41586-020-2649-2)).

- Scikit-learn
  - F.   Pedregosa,   G.   Varoquaux,   A.   Gramfort,   V.   Michel,B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss,V.   Dubourg,   J.   Vanderplas,   A.   Passos,   D.   Cournapeau,M.  Brucher,  M.  Perrot,  and  E.  Duchesnay,  “Scikit-learn:Machine  learning  in  Python,”Journal  of  Machine  LearningResearch, vol. 12, pp. 2825–2830, 2011.

- PyTorch
  - Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G.,  Killeen, T.,  Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J.J., Chintala, S., “PyTorch: An Imperative Style, High-Performance Deep Learning Library”, Advances in Neural Information Processing Systems 32, pp 8024--8035, 2019. [Publication link](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)

- Transformers

  - [GitHub Repo](https://github.com/huggingface/transformers)


  - Wolf, Thomas, et al. "HuggingFace's Transformers: State-of-the-art Natural Language Processing." *ArXiv* (2019): arXiv-1910.
  - [Tutorial and Documentation](https://huggingface.co/transformers/index.html)

- Tokenizers

  - [GitHub Repo](https://github.com/huggingface/tokenizers)

- [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)

- [Kaggle Competition: Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction/overview)

- [Notebook solutions on Kaggle Competition: Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction/notebooks)

