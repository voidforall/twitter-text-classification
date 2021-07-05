from dataloader.TweetReader import TweetReader
from predictor import TextClassifierPredictor
from models.basic_classifier import BasicClassifier
from configs.defaults import get_cfg_defaults

import os
import sys
import json
import numpy as np
from overrides import overrides
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data.batch import Batch
from allennlp.common import Params

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper, LstmSeq2VecEncoder,CnnEncoder
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder, PytorchTransformer
from allennlp.modules.attention.attention import Attention
from allennlp.training import GradientDescentTrainer, TensorboardWriter
from allennlp.nn import InitializerApplicator

import sys

if __name__ == '__main__':

	# Load configurations, specify in `merge_from_file(config_path)`
	# cfg = get_cfg_defaults()
	# cfg.merge_from_file("configs/{}.yaml".format(sys.argv[1]))
	# cfg.freeze()
	cfg = get_cfg_defaults()

	import argparse
	parser = argparse.ArgumentParser(description='Machine Learning Course Project 2, commandline parameter')
	parser.add_argument('--yaml', dest = "yaml", type=str, default=None, help="experiment yaml setting", required=True)
	args = parser.parse_args()

	if "yaml" not in args.yaml:
		cfg.merge_from_file("configs/{}.yaml".format(args.yaml))
	else:
		cfg.merge_from_file("configs/{}".format(args.yaml))

	cfg.freeze()

	# Specify the device ID (not sure when running on the cluster)
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	print("EXP %s evaluation starts" % cfg.EXP.NAME)

	# read train and valid data
	if cfg.DATA.SENTENCE_SEGMENTATION == True:
		reader = TweetReader(segment_sentences=True, max_sentence_num=16, max_sequence_length=64) # FIXME: 16 should be reconsidered
	else:
		reader = TweetReader(segment_sentences=False, max_sequence_length=64)
	# reader = TweetReader(segment_sentences=False, max_sequence_length=64)
	vocab = Vocabulary.from_files(cfg.PATH.VOCAB_FILE)
	print("[STATUS] Vocabulary loaded.")

	# -------------------------------------------
	# Set the model architecture
	# -------------------------------------------
	# The model architectures have both the similarities and differences.
	# Shared part: 
	# 		embedding_layer: acquire the word embeddings, can be trained from scratch or pre-trained
	# Specified part: 
	# 		encoder: from word embedding to `tweet embedding`, could be RNN/CNN/RCNN/Transformer/HAN etc.
	# 		decoder: from `tweet embedding` to `sentiment prediction`, could be hidden Linear layers and the output layer.
	# -------------------------------------------

	sent_enc_input = cfg.ARCH.SENTENCE_ENCODER.INPUT_DIM
	sent_enc_hidden = cfg.ARCH.SENTENCE_ENCODER.HIDDEN_DIM
	sent_enc_nlayer = cfg.ARCH.SENTENCE_ENCODER.NUM_LAYER
	sen_enc_nfilter = cfg.ARCH.SENTENCE_ENCODER.FILTER_NUM
	sen_enc_noutput = cfg.ARCH.SENTENCE_ENCODER.OUTPUT_DIM
	sen_enc_dropout = cfg.ARCH.SENTENCE_ENCODER.DROPOUT
	sen_enc_headnum = cfg.ARCH.SENTENCE_ENCODER.TRANSFORMER_NHEAD 
	feed_input = cfg.FEED.INPUT_SIZE
	feed_layers = cfg.FEED.LAYERS
	feed_hidden = cfg.FEED.HIDDEN
	feed_dropout = cfg.FEED.DROPOUT2

	if cfg.MODEL.NAME == "RCNNClassifier":
		conv_layer = torch.nn.Conv1d(2 * sent_enc_hidden+sent_enc_input, feed_input,1)
	else:
		conv_layer = None
		
	# Word Embedding layer ()
	emb_params = Params(
		{
			"token_embedders": {
				"tokens": {
					"type": "embedding", 
					"embedding_dim": cfg.ARCH.EMBEDDING.DIM,
					"pretrained_file": cfg.PATH.PRETRAINED_EMBEDDING_FILE, 
					"trainable": True,
				}
			}
		}
	)
	embedding_layer = BasicTextFieldEmbedder.from_params(vocab=vocab, params=emb_params)

	# Initializer
	initializer = InitializerApplicator()

	# Classifiers and their settings
	if cfg.MODEL.NAME == "RNNClassifier":
		seq2vec_encoder = LstmSeq2VecEncoder(
			input_size=sent_enc_input,
			hidden_size=sent_enc_hidden,
			num_layers=sent_enc_nlayer,
			bidirectional=True,
			dropout=sen_enc_dropout
		)
	elif cfg.MODEL.NAME == "CNNClassifier":
		seq2vec_encoder = CnnEncoder(
			embedding_dim=sent_enc_input, 
			num_filters=sen_enc_nfilter, 
			output_dim=feed_input
		)
	elif cfg.MODEL.NAME == "RCNNClassifier":
		seq2vec_encoder = LstmSeq2SeqEncoder(
			input_size=sent_enc_input,
			hidden_size=sent_enc_hidden,
			num_layers=sent_enc_nlayer,
			bidirectional=True,
			dropout=sen_enc_dropout
		)
	elif cfg.MODEL.NAME == "TransformerClassifier":
		seq2vec_encoder = PytorchTransformer(
			input_dim=sent_enc_input,
			num_layers=sent_enc_nlayer,
			feedforward_hidden_dim=feed_input,
			num_attention_heads=sen_enc_headnum,
			positional_embedding_size=sent_enc_input,
			dropout_prob=sen_enc_dropout,
			activation= "relu"
		)
	elif cfg.MODEL.NAME == "HAN":
		seq2vec_encoder = None
	else:
		print('invalid Classifier!')

	feedforward = FeedForward(
		input_dim=feed_input,
		num_layers=feed_layers,
		hidden_dims=feed_hidden,
		activations=torch.nn.ReLU(),
		dropout=feed_dropout
	)

	# Instantialize the model. It will call different model api given the model configuration
	if cfg.MODEL.NAME != "HAN":
		model = BasicClassifier(
			vocab=vocab,
			text_field_embedder=embedding_layer,
			seq2vec_encoder=seq2vec_encoder,
			feedforward=feedforward,
			label_namespace="label",
			namespace="tokens",
			initializer=initializer,
			conv_layer=conv_layer,
			modelname=cfg.MODEL.NAME,
			mode="eval"
		)
	else: 
		from models.HAN_classifier import HAN
		model = HAN(
			vocab=vocab,
			text_field_embedder=embedding_layer,
			feedforward=feedforward,
			label_namespace="label",
			namespace="tokens",
			initializer=initializer,
			word_encoder_embedding_size = cfg.ARCH.EMBEDDING.DIM,
			word_gru_hidden = cfg.ARCH.HAN.GRU_HIDDEN,
			sent_gru_hidden = cfg.ARCH.HAN.GRU_HIDDEN,#,
			bidirectional = True, 
		)

	# Load model weights
	print("[STATUS] Loading the trained weights into the model.")
	weights_file_path = os.path.join(cfg.PATH.LOG_DIR, "best.th")
	with open(weights_file_path, "rb") as f:
		model.load_state_dict(torch.load(f))

	# The model has to be on the correct device
	if torch.cuda.is_available():
		cuda_device = cfg.SYSTEM.DEVICE_ID
	else:
		cuda_device = -1

	model = model.cuda(cuda_device)
	print("[STATUS] Model has been setup.")

	# Set the evaluation
	# 1. Evaluate the performance on test set.
	# 2. Predict the results on predict set, and output as submission file.
	test_file_path = cfg.PATH.TEST_FILE
	with open(test_file_path, "r", encoding="utf-8") as test_file:
		testset_json = json.load(test_file)
	test_texts = [v["text"] for v in testset_json]
	test_labels = [v["label"] for v in testset_json]

	predictor = TextClassifierPredictor(model, dataset_reader=reader, frozen=True)
	prediction_list = []
	for text in test_texts:
		probs = predictor.predict(text)["probs"]
		label = np.argmax(probs)
		prediction_list.append(label)

	from sklearn.metrics import accuracy_score
	from sklearn.metrics import f1_score
	print('Calculating F1 and acc for test set.')
	acc = accuracy_score(test_labels, prediction_list)
	f1 = f1_score(test_labels, prediction_list)
	print('Total acc: %.4f' % acc)
	print('Total f1: %.4f' % f1)

	predict_file_path = cfg.PREDICT.PREDICT_FILE
	with open(predict_file_path, "r", encoding="utf-8") as predict_file:
		predict_json = json.load(predict_file)
	predict_texts = [v["text"] for v in predict_json]
	prediction_list = []
	for text in predict_texts:
		probs = predictor.predict(text)["probs"]
		label = np.argmax(probs)
		prediction_list.append(label)

	labels = [label if label==1 else -1 for label in prediction_list]
	ids = list(range(1, len(labels) + 1))

	from utils.result_helpers import create_csv_submission
	output_file_path = os.path.join(cfg.PATH.LOG_DIR, "submission.csv")
	create_csv_submission(ids, labels, output_file_path)

	print("[STATUS] Evaluation ends, all results have been written into %s" % output_file_path)
