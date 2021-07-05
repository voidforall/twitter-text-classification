# Project 2 Text Classification

> Code has been tested on Linux Ubuntu

### Requirements

```
python=3.6.5
torch==1.7.0
allennlp==1.2.1
yacs==0.1.6
sklearn
```

Please put the following data files into the `data` folder

- `test_data.txt`
- `train_neg.txt`
- `train_neg_full.txt`
- `train_pos.txt`
- `train_pos_full.txt`

### Directory Tree

```
│   README.md
│   requirements.txt
│   train.py
│   evaluate.py
│   predictor.py
│
├───data
│       glove.twitter.27B.200d.txt
│       test_data.txt
│       train_neg.txt
│       train_neg_full.txt
│       train_pos.txt
│       train_pos_full.txt
│
├── scripts
|       preprocess.py
|       split_dataset.py
|       txt_to_json.py
|   
├── configs
│       defaults.py
│       CNN.yaml
│       RNN.yaml
│       RCNN.yaml
│       Transformer.yaml
│       HANFULL256.yaml
│       CNN_sub.yaml
│       RNN_sub.yaml
│       RCNN_sub.yaml
│       Transformer_sub.yaml
│       HAN256.yaml
│
├── dataloader
|       reader_test.py
│       TweetReader.py
│
├── models
|       basic_classifier.py
└       HAN_classifier.py


```

## File Description

- `requirements.txt`
  - The required python packages and versions.
- `train.py`
  - Scripts for training deep-learning models: CNN, RNN, RCNN, Transformer and HAN.
- `evaluate.py`
  - Scripts for evaluating the deep-learning models.
- `data`
  - The data folder, contains the data set for training and submission.
- `scripts`
  - The folder for data cleaning and splitting scripts.
- `configs`
  - The configuration folder, contains the default parameters and the specified parameters for all deep-learning models.
- `dataloader`
  - The folder for data loading scripts that are used in train.py.
- `models`
  - The folder for deep-learning model scripts that are used in train.py. basic_classifier.py contains CNN, RNN and RCNN, HAN_classifier.py contains HAN.
- `utils`
  - The folder that contains the scripts for submission file creation.
    

## HOW-TO USE

### To get the GloVe for twitter:

```shell
:~/deep-pipeline/data$ wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
:~/deep-pipeline/data$ unzip glove.twitter.27B.zip
```
Download from [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip) and unzip the file. Move the file `glove.twitter.27B.200d.txt` to `data` directory

### To preprocess the data

```shell
:~/deep-pipeline/scripts$ python preprocess.py --dataset $data_type
```
`$data_type` should be `subset`, `fullset` or `test`, representing the cleaning for the subset, the full data set or the submission data set respectively.

#### What to expect?

The cleaned data set `~/deep-pipeline/data/*_clean.txt`.

### To split the data to a train set, a validation set and a test set for the model training

```shell
:~/deep-pipeline/scripts$ python split_dataset.py --dataset $data_type
```
`$data_type` should be `subset` or `fullset`, meaning that you are splitting the subset or the full data set.

#### What to expect?

`~/deep-pipeline/data/train_*.json` and `~/deep-pipeline/data/valid_*.json` for the specified data type, they are used while training. They are the same as the data provided in the `README` file in the parent directory.

### To convert the submission data into .json file

```shell
:~/deep-pipeline/scripts$ python txt_to_json.py
```

#### What to expect?

`~/deep-pipeline/data/predict.json` used in evaluation.

### To train the deep-learning models

```shell
:~/deep-pipeline$ python train.py --yaml $config_name
```
`$config_name` should be `CNN`, `RNN`, `RCNN`, `Transformer` or `HANFULL256` for CNN, RNN, RCNN, Transformer and HAN models trained on the full data sets, and `CNN_sub`, `RNN_sub`, `RCNN_sub`, `Transformer_sub` or `HAN256` are models trained on subsets.

#### What to expect?

Progress of instance reading, vocabulary building, data reading and model training will be presented. The training outputs are saved in `~/deep-pipeline/saved`.

### To evaluate the deep-learning models

```shell
:~/deep-pipeline$ python evaluate.py --yaml $config_name
```
`$config_name` should be `CNN`, `RNN`, `RCNN`, `Transformer`, `HANFULL256`, `CNN_sub`, `RNN_sub`, `RCNN_sub`, `Transformer_sub`, `HAN256`. The evaluation of models trained on subsets should follow its training to have the same vocabulary dictionary length.

#### What to expect?

The accuracy and f1 score of the specified model's name, the submission file is saved as `~/deep-pipeline/saved/$config_name/submission.csv`.


## HOW-TO Replicate

To prepare the data set (`$data_type`) and get the accuracy and f1 score of one deep-learning model (`$config_name`) trained on it, we execute
```shell
:~/deep-pipeline/scripts$ python preprocess.py --dataset $data_type
:~/deep-pipeline/scripts$ python split_dataset.py --dataset $data_type
:~/deep-pipeline/scripts$ python txt_to_json.py
:~/deep-pipeline/scripts$ cd ..
:~/deep-pipeline$ python train.py --yaml $config_name
:~/deep-pipeline$ python evaluate.py --yaml $config_name
```
For example, to get the prediction and evaluation of our RCNN model trained on the full data set, execute
```shell
:~/deep-pipeline/scripts$ python preprocess.py --dataset fullset
:~/deep-pipeline/scripts$ python split_dataset.py --dataset fullset
:~/deep-pipeline/scripts$ python txt_to_json.py
:~/deep-pipeline/scripts$ cd ..
:~/deep-pipeline$ python train.py --yaml RCNN
:~/deep-pipeline$ python evaluate.py --yaml RCNN
```


## References
- NumPy
  - Harris, C.R., Millman, K.J., van der Walt, S.J. et al. *Array programming with NumPy*. Nature 585, 357–362 (2020). DOI: [0.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2). ([Publisher link](https://www.nature.com/articles/s41586-020-2649-2)).
- Scikit-learn
  - F.   Pedregosa,   G.   Varoquaux,   A.   Gramfort,   V.   Michel,B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss,V.   Dubourg,   J.   Vanderplas,   A.   Passos,   D.   Cournapeau,M.  Brucher,  M.  Perrot,  and  E.  Duchesnay,  “Scikit-learn:Machine  learning  in  Python,”Journal  of  Machine  LearningResearch, vol. 12, pp. 2825–2830, 2011.
- Overrides
  - [Avalable link](https://pypi.org/project/overrides/)
- GloVe
  - Pennington, J., Socher, R., & Manning, C. D. (2014, October). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).
- Allennlp
  - Gardner, Matt, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson H S Liu, Matthew E. Peters, Michael Schmitz and Luke Zettlemoyer. “A Deep Semantic Natural Language Processing Platform.” (2017).
- PyTorch
  - Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G.,  Killeen, T.,  Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J.J., Chintala, S., “PyTorch: An Imperative Style, High-Performance Deep Learning Library”, Advances in Neural Information Processing Systems 32, pp 8024--8035, 2019. [Publication link](http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)
