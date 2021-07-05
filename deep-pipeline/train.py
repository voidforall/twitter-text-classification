from dataloader.TweetReader import TweetReader
from models.basic_classifier import BasicClassifier
from configs.defaults import get_cfg_defaults

import os
import sys
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
	# -------------------------------------------
	# Basic configurations
	# -------------------------------------------
	# Load configurations, specify in `merge_from_file(config_path)`
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

	# Fix the random seed to do consistent experiments
	np.random.seed(42)
	torch.manual_seed(42)
	torch.cuda.manual_seed_all(42)

	# Specify the device ID (not sure when running on the cluster)
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	# -------------------------------------------
	# Load dataset
	# -------------------------------------------
	# TweetReader:
	# 		segment_sentences: bool, returns List[Text] when it is True (for Hierarchical model)
	# 		max_sentence_num: int, only use sentence segmentation in hierarchical setting
	# 		max_sequence_length: int, truncate the data to the maximum length
	# vocab:
	# 		min_count: int, choose the minimum occuring times of a word in the dataset,
	# 						others will be replaced as <UNK>
	# 		pretrained_files: file path, required when using pretrained word embedding
	# -------------------------------------------
	print("EXP %s starts" % cfg.EXP.NAME)


	# read train and valid data
	if cfg.DATA.SENTENCE_SEGMENTATION == True:
		reader = TweetReader(segment_sentences=True, max_sentence_num=16, max_sequence_length=64) 
	else:
		reader = TweetReader(segment_sentences=False, max_sequence_length=64)
	
	
	# read train and valid data
	print("[PROCESSING] Loading training set and validation set...")
	train_dataset = reader.read(cfg.PATH.TRAIN_FILE)
	valid_dataset = reader.read(cfg.PATH.VALID_FILE)

	# Set vocabulary
	if cfg.MODEL.EMBEDDING_PRETRAINED:
		pretrained_w2v_path = cfg.PATH.PRETRAINED_EMBEDDING_FILE
		vocab = Vocabulary.from_instances(instances=train_dataset+valid_dataset,
										  min_count={"tokens":cfg.DATA.VOCAB_MIN_COUNT},
										  pretrained_files={"tokens":pretrained_w2v_path})
	else:
		vocab = Vocabulary.from_instances(instances=train_dataset+valid_dataset,
										  min_count={"tokens":cfg.DATA.VOCAB_MIN_COUNT})
	vocab.save_to_files(cfg.PATH.VOCAB_FILE)
	print("[STATUS] Dataset loaded.")

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
			mode = "train"
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
			sent_gru_hidden = cfg.ARCH.HAN.GRU_HIDDEN,
			bidirectional = True,
		)

	# The model has to be on the correct device before trainer.train()
	if torch.cuda.is_available():
		cuda_device = cfg.SYSTEM.DEVICE_ID
	else:
		cuda_device = -1
	model = model.cuda(cuda_device)
	print("[STATUS] Model has been setup.")

	# -------------------------------------------
	# Set training hyperparameters
	# -------------------------------------------
	# dataloader: load the dataset, with specified batching scheme
	# 	
	# optimizer: optimize the model weights, many choices in type (Adam, Adagrad, SGD etc.) 
	# 	and hyperparameters (learning rate, regularization etc.)
	# 	
	# trainer: serves for the entire training process (logging, validation etc.)
	# 	
	# -------------------------------------------
	train_dataset = AllennlpDataset(train_dataset, vocab=vocab)
	valid_dataset = AllennlpDataset(valid_dataset, vocab=vocab)
	train_dataset_loader = PyTorchDataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
	valid_dataset_loader = PyTorchDataLoader(valid_dataset, batch_size=cfg.TRAIN.BATCH_SIZE)

	# Set optimizer
	optimizer = optim.Adam(
		model.parameters(), 
		lr=cfg.TRAIN.LR,
		betas=(0.9, 0.999), 
		eps=cfg.TRAIN.EPS,
		weight_decay=cfg.TRAIN.L2_NORM
	)

	# Set trainer
	trainer = GradientDescentTrainer(
		model=model,
		optimizer=optimizer,
		data_loader=train_dataset_loader,
		patience=cfg.TRAIN.PATIENCE,
		validation_metric="-loss",
		validation_data_loader=valid_dataset_loader,
		num_epochs=cfg.TRAIN.NUM_EPOCHS,
		serialization_dir=cfg.PATH.LOG_DIR,
		checkpointer=None,
		cuda_device=cuda_device,
		grad_norm=cfg.TRAIN.GRAD_NORM,
		grad_clipping=cfg.TRAIN.GRAD_CLIPPING,
		learning_rate_scheduler=None,
		momentum_scheduler=None,
		moving_average=None,
		batch_callbacks=None,
		epoch_callbacks=None
	)
	print("[STATUS] Trainer has been setup.")


	print("[PROCESSING] Training starts, results will be serialized to specified directory.")
	trainer.train()
