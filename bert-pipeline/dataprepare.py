import os
import re
import json
import torch
import numpy as np
import random
from transformers import *

"""
	This script provides some data preparation functions for BERT-based model training.
"""

def bert_tokenize(dataset, tokenizer, max_len, is_ex=False):
	"""
		Apply BERT Tokenizer, tokenize the original sequence and map tokens to their IDs.
		Padding and Truncation will be performed according to `max_len`.
	"""
	input_ids = []
	attn_masks = []
	for sample in dataset:
		text = sample[0]
		if is_ex == True:
			text = sample[2]
		encoded_dict = tokenizer.encode_plus(
							text,
							add_special_tokens = True,
							truncation = True,
							max_length = max_len,
							padding = 'max_length',
							return_attention_mask = True,
							return_tensors = 'pt',
						)
		input_ids.append(encoded_dict["input_ids"])
		attn_masks.append(encoded_dict["attention_mask"])
	# Convert lists to tensors
	input_ids = torch.cat(input_ids, dim=0)
	attn_masks = torch.cat(attn_masks, dim=0)
	return input_ids, attn_masks

def bert_tokenize_ex(dataset, tokenizer, max_len):
	"""
		The above bert-tokenize code for the extraction-augmented data.
	"""
	input_ids = []
	attn_masks = []
	token_type_ids = []
	for sample in dataset:
		text = sample[0]
		extracted = sample[2]
		encoded_dict = tokenizer.encode_plus(
							text,
							text_pair = extracted,
							add_special_tokens = True,
							truncation = True,
							max_length = max_len,
							padding = 'max_length',
							return_attention_mask = True,
							return_tensors = 'pt',
						)
		input_ids.append(encoded_dict["input_ids"])
		attn_masks.append(encoded_dict["attention_mask"])
		token_type_ids.append(encoded_dict["token_type_ids"])

	# Convert lists to tensors
	input_ids = torch.cat(input_ids, dim=0)
	attn_masks = torch.cat(attn_masks, dim=0)
	token_type_ids = torch.cat(token_type_ids, dim=0)
	return input_ids, attn_masks, token_type_ids

# Ref: https://github.com/AntiDeprime/GloVeTwitterPreprocPy/blob/master/GloVePreprocessor.py
class GloVePreprocessor(object):
	'''Tweet preporcessor for glove.twitter.27B
	
		:param lowercase: Transform tweets to lowercase?, defaults to True
		:type lowercase: Bool
		
	'''
	
	def __init__(self, lowercase = True):   
		self.lowercase = lowercase
	
	def __hashtags__ (self, hashtag):    
		hashtag_body = hashtag.group()[1:]
		if( hashtag_body.upper == hashtag_body):
			result = f' <hashtag> {hashtag_body} <allcaps> '
		else:
			result = ' <hashtag> ' + ' '.join(hashtag_body.split('/(?=[A-Z])/)'))
		return (result)       

	def preprocess (self, tweet):
		'''Preprocessor function
		
		:param tweet: Tweet string
		:type tweet: str
		:returns: Preprocessed Tweet 
		:rtype: str
		
		'''
		self.tweet = tweet

		# Different regex parts for smiley faces
		eyes = '[8:=;]'
		nose = "['`\-]?"
		
		# Mark allcaps words
		self.tweet = re.sub(r'\b[A-Z][A-Z0-9]+\b', 
					lambda x: f'{x.group().lower()} <allcaps> ', 
					self.tweet)
		
		# Mark urls
		self.tweet = re.sub('https?:\/\/\S+\b|www\.(\w+\.)+\S*', " <url> ", self.tweet)
		# Force splitting words appended with slashes (once we tokenized the URLs, of course)
		self.tweet = re.sub('/',' / ', self.tweet) 
		# Mark @users
		self.tweet = re.sub('@\w+', ' <user> ', self.tweet)

		# Mark smileys
		self.tweet = re.sub(f'{eyes}{nose}[)dD]+|[)dD]+{nose}{eyes}', ' <smile> ', self.tweet)
		self.tweet = re.sub(f'{eyes}{nose}[pP]+', '<lolface>', self.tweet)
		self.tweet = re.sub(f'{eyes}{nose}\(+|\)+{nose}{eyes}', ' <sadface> ', self.tweet)
		self.tweet = re.sub(f'{eyes}{nose}[\/|l*]', ' <neutralface> ', self.tweet)
		self.tweet = re.sub('<3',' <heart> ', self.tweet)
		
		# Mark numbers 
		self.tweet = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' <number> ', self.tweet)

		# Mark hashtags 
		self.tweet = re.sub('#\S+', self.__hashtags__, self.tweet)

		# Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
		self.tweet = re.sub('([!?.]){2,}', 
					lambda x: f'{x.group()[0]} <repeat> ', 
					self.tweet)

		# Mark elongated words like heyyy -> hey <elong>
		self.tweet = re.sub(r'\b(\S*?)(.)\2{2,}\b', lambda x: f'{x.group(1)}{x.group(2)} <elong> ', self.tweet)
		
		# To lowercase 
		if self.lowercase:
			self.tweet = self.tweet.lower()
		
		# Trim whitespaces 
		self.tweet = ' '.join(self.tweet.split())

		return (self.tweet)


def process_extract_csv(tweet, selected_text, sentiment, tokenizer, max_len):
	"""
	Preprocess tweet sentiment extraction data.
	"""
	len_st = len(selected_text)
	idx0 = None
	idx1 = None
	for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
		if tweet[ind: ind+len_st] == selected_text:
			idx0 = ind
			idx1 = ind + len_st - 1
			break
	
	char_targets = [0] * len(tweet)
	if idx0 != None and idx1 != None:
		for ct in range(idx0, idx1 + 1):
			char_targets[ct] = 1
	
	tok_tweet = tokenizer.encode(tweet)
	input_ids_orig = tok_tweet.ids[1:-1]
	tweet_offsets = tok_tweet.offsets[1:-1]
	
	target_idx = []
	for j, (offset1, offset2) in enumerate(tweet_offsets):
		if sum(char_targets[offset1: offset2]) > 0:
			target_idx.append(j)
	
	if len(target_idx) == 0:
		return None
	
	targets_start = target_idx[0] + 1
	targets_end = target_idx[-1] + 1

	input_ids = tok_tweet.ids
	mask = [1] * len(input_ids)
	tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)]

	padding_length = max_len - len(input_ids)
	if padding_length > 0:
		input_ids = input_ids + ([0] * padding_length)
		mask = mask + ([0] * padding_length)
		tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
	
	if sentiment == "negative":
		sentiment_label = 0
	else:
		sentiment_label = 1

	return {
		'ids': input_ids,
		'mask': mask,
		'targets_start': targets_start,
		'targets_end': targets_end,
		'sentiment': sentiment_label,
		'tweet_offsets': tweet_offsets,
		'tweet_text': tweet
	}

def process_test_data(tweet, tokenizer, max_len):
	"""
	Preprocess tweet sentiment extraction data when doing inference.
	"""
	tok_tweet = tokenizer.encode(tweet)
	input_ids_orig = tok_tweet.ids[1:-1]
	tweet_offsets = tok_tweet.offsets[1:-1]
		
	input_ids = tok_tweet.ids
	mask = [1] * len(input_ids)
	tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)]

	padding_length = max_len - len(input_ids)
	if padding_length > 0:
		input_ids = input_ids + ([0] * padding_length)
		mask = mask + ([0] * padding_length)
		tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)

	return {
		'ids': input_ids,
		'mask': mask,
		'tweet_offsets': tweet_offsets,
		'tweet_text': tweet
	}
