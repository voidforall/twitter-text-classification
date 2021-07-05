# Machine Learning Project 2: Tweet Sentiment Analysis 

## Overview

Project codes are grouped into four categories: basic-pipeline, deep-pipeline, bert-pipeline and visualization. They are four directories in this project code. The codes are tested on Linux Platform.

- Basic-pipeline
  - Logistic Regression
  - Support Vector Machine
- Deep-pipeline
  - TextCNN
  - TextRNN
  - RCNN
  - Transformer
  - Hierarchical Attention Networks  (HAN)
- BERT-pipeline
  - BERT
  - RoBERTa
  - BERT Sentiment Extraction
  - BERT Ensemble

> Detailed HOW-TO, References and documentations are in sub-folders. 

## Shortest-path to evaluate fine-tuned RoBERTa model

Notes: This section provides the steps to replicate our results. The result is from fine-tuned RoBERTa and its performance may be a little lower than our ensembled results on AICrowd Platform. We provide the best performance model in `run.py` other than the ensembled model, for the difficulty to include all ensemble constituents in one script.

### 1. Prepare the data

First, you need to prepare the data as we used in the experiments. We offered two options: (1) Following the README in deep-pipeline, use the scripts to pre-process, split the dataset, or (2) directly download data from google drive.

Since we also prepare the sentiment extraction augmented data in the google drive, we recommend to use option 2 here. To directly download the prepared dataset, use the link: https://drive.google.com/file/d/1mKpBDYOJmykhpu6QvekFPNap-Ex0baOG/view?usp=sharing

### 2. Download the RoBERTa pretrained weights 

We also upload our pretrained weights of RoBERTa to google drive, you can download from : https://drive.google.com/file/d/1N618fh25nBpVZ2N3d9d1zKftG0JLAP-b/view?usp=sharing

Then, unzip the file and put it under folder `./bert-pipeline/roberta_weights`

### 3. Run the evaluation script and get the results

Execute the evaulation script with `python  run.py`, it will output the performance on test set and save the predictions in `./bert-pipeline/roberta_weights/submission.csv`.

## To replicate the best result on AICrowd

Our best result is derived from the ensembled models, and it may take a lot of time to train them all. Hence, we provide both the pre-trained weights and their pre-evaluated results in google drive: https://drive.google.com/file/d/1CaH1RHL7lR-lBCqrB_rOTtUQOEu0A0gV/view?usp=sharing

To replicate the best result, unzip the file and save the model log folders in `./bert-pipeline/saved`, then run `python ensemble.py` to get the ensembled results. `./bert-pipeline/submission.csv` is **exactly the same with our best result on AICrowd platform**.

If you want to replicate the evaluation process, follow the guideline in Bert-pipeline README file.

## File Tree

```
.
└── EPFL_ML_project2
    ├── basic-pipeline
    │   ├── build_voca_to_glove.py
    │   ├── data_utils.py
    │   ├── framework.py
    │   ├── run.py
    │   └── README.md
    ├── bert-pipeline
    │   ├── bert_ensemble.py
    │   ├── bert_evaluate_exaug.py
    │   ├── bert_evaluate_exsiamense.py
    │   ├── bert_evaluate.py
    │   ├── bert_extract_infer.py
    │   ├── bert_extract_train.py
    │   ├── bert_train_exaug.py
    │   ├── bert_train_exsiamense.py
    │   ├── bert_train.py
    │   ├── dataprepare.py
    │   ├── preprocess.py
    │   ├── roberta_evaluate.py
    │   └── roberta_train.py
    ├── deep-pipeline
    │   ├── configs
    │   │   ├── ...
    │   ├── data
    │   │   ├── ...
    │   ├── dataloader
    │   │   ├── reader_test.py
    │   │   └── TweetReader.py
    │   ├── evaluate.py
    │   ├── models
    │   │   ├── basic_classifier.py
    │   │   └── HAN_classifier.py
    │   ├── predictor.py
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── scripts
    │   │   ├── preprocess.py
    │   │   ├── split_dataset.py
    │   │   └── txt_to_json.py
    │   ├── train.py
    │   └── utils
    │       └── result_helpers.py
    ├── README.md
    └── visualization
        ├── loss_accuracy-plot
        │   ├── accuracy_validation.png
        │   ├── loss_acc-plots.py
        │   ├── loss_accuracy.png
        │   ├── loss_train.png
        │   └── ...csv
        └── wordcloud
            ├── frequency.py
            └── upvote.png
```

