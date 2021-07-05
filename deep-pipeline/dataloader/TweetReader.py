import logging
from typing import List, Dict
from overrides import overrides

import re
import json
import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ListField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.vocabulary import DEFAULT_PADDING_TOKEN

logger = logging.getLogger(__name__)

class TweetReader(DatasetReader):
	"""
	Read tokens and labels from a tweet sentiment classification dataset.

	Expected data: List[Instance], each instance consists two fields: 
		tokens: TextField (or ListField[TextField]), label: LabelField  

	Ref: https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/text_classification_json.py
	"""

	def __init__(
		self,
		tokenizer: Tokenizer = None,
		token_indexers: Dict[str, Tokenizer] = None,
		segment_sentences: bool = False,
		max_sentence_num: int = None,
		max_sequence_length: int = None,
		lazy: bool = False
	) -> None:
		super().__init__(lazy)
		self._tokenizer = tokenizer or WhitespaceTokenizer()
		self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
		self._segment_sentences = segment_sentences
		self._max_sentence_num = max_sentence_num
		self._max_sequence_length = max_sequence_length
		self._min_sequence_length = 5 # which is CNN kernel size
		if self._segment_sentences:
			self._sentence_splitter = SpacySentenceSplitter(rule_based=True) # use punctuation to detect boundaries

	@overrides
	def _read(self, file_path: str):
		""" Read a given tweet sentiment classification dataset. """
		with open(cached_path(file_path), "r", encoding="utf-8") as data_file:
			logger.info("Reading data from file at %s", file_path)
			data_list = json.load(data_file)

			for sample in data_list:
				text = sample["text"].strip()
				if "label" not in sample.keys():
					yield self.text_to_instance(text)
				else:
					yield self.text_to_instance(text, sample["label"])

	def _truncate(self, tokens):
		""" Truncate tokens to a given sequence length. """
		if len(tokens) > self._max_sequence_length:
			tokens = tokens[:self._max_sequence_length]
		return tokens

	@staticmethod 
	def _tokens_to_ids(tokens: List[Token]) -> List[int]:
		ids: Dict[str, int] = {}
		out: List[int] = []
		for token in tokens:
			out.append(ids.setdefault(token.text, len(ids)))
		return out

	@overrides
	def text_to_instance(self, text: str, label: int = None) -> Instance:
		""" Form the data to Instance. """
		fields: Dict[str, Field] = {}

		# process the text to ListField (sentence-level) or TextField (token-level)
		if self._segment_sentences:
			sentences = []
			text_splitted = self._sentence_splitter.split_sentences(text)
			# truncate sentences
			if self._max_sentence_num is not None and len(text_splitted) > self._max_sentence_num:
				text_splitted = text_splitted[:self._max_sentence_num]

			for sentence in text_splitted:
				tokens = self._tokenizer.tokenize(sentence)
				# truncate sequences
				if self._max_sequence_length is not None:
					tokens = self._truncate(tokens)
				sentences.append(TextField(tokens, self._token_indexers))
			fields["tokens"] = ListField(sentences)

		else:
			tokens = self._tokenizer.tokenize(text)
			if self._max_sequence_length is not None:
				tokens = self._truncate(tokens)
			while len(tokens) < self._min_sequence_length:
				tokens.append(Token(DEFAULT_PADDING_TOKEN))
			fields["tokens"] = TextField(tokens, self._token_indexers)

		if label is not None:
			fields["label"] = LabelField(label, skip_indexing=True)
		return Instance(fields)