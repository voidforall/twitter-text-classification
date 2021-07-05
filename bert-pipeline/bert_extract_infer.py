import os
import json
import time
import datetime
import torch
import numpy as np
import pandas as pd
import tokenizers
import string
import nltk
from transformers import *
import torch.nn as nn
import dataprepare

print("BERT Extract Inference starts!")

# Fix random seeds and specify GPU device
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Load test data / predict data
inference_datapath = r"./data/train_full.json"

with open(inference_datapath, "r", encoding="utf-8") as f:
	infer_list = json.load(f)

infer_set = [sample["text"].strip() for sample in infer_list]

output_dir = r"./saved/bert-extract"

# Use Bert tokenizer to re-tokenize (WordPiece)
tokenizer = tokenizers.BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
tokenizer.add_special_tokens(["<allcaps>", "<elong>", "<url>", "<smile>", "<lolface>", "<sadface>", "<neutralface>", "<heart>", "<number>", "<repeat>"])

# Extract the required fields of data (input_ids, masks, targets_start, targets_end, sentiment)
max_len = 200
dataset = []
for i in range(len(infer_set)):
	output_dict = dataprepare.process_test_data(infer_set[i], tokenizer, max_len)
	dataset.append((output_dict["ids"], output_dict["mask"], output_dict["tweet_offsets"], output_dict["tweet_text"]))

infer_inputs, infer_masks = torch.tensor([i[0] for i in dataset]), torch.tensor([i[1] for i in dataset])
infer_offsets = torch.tensor([i[2] for i in dataset])
infer_index = torch.tensor([i for i in range(len(dataset))])

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
batch_size = 32     # author recommend batch size of 16 or 32 for fine-tuning BERT
# Create the DataLoader
infer_data = TensorDataset(infer_inputs, infer_masks, infer_offsets, infer_index)
infer_dataloader = DataLoader(
		infer_data, 
		sampler=SequentialSampler(infer_data), 
		batch_size=batch_size
	)

# Declare the model structure
class BertForExtraction(BertPreTrainedModel):
	"""
		This module is composed of BERT-base-uncased with a
		linear layer on top of the pooled output, specific
		for downstream task - sentiment extraction
	"""
	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(0.1)
		self.extractor = nn.Linear(config.hidden_size * 2, 2)

		self.init_weights()
	
	def forward(
		self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None, 
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		start=None,
		end=None
	):
	
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_hidden_states=True
		)
		
		hidden_states = outputs[2]  # (batch_size, sequence_length, hidden_size)
		out = torch.cat((hidden_states[-1], hidden_states[-2]), dim=-1)
		out = self.dropout(out) # shape: (batch_num, hidden_size*2)
		logits = self.extractor(out) # shape: (batch_num, 2)
		
		start_logits, end_logits = logits.split(1, dim=-1)
		start_logits = start_logits.squeeze(-1)
		end_logits = end_logits.squeeze(-1)
		
		ret_outputs = start_logits, end_logits

		if start is not None and end is not None:
			loss_fct = torch.nn.CrossEntropyLoss()
			start_loss = loss_fct(start_logits, start)
			end_loss = loss_fct(end_logits, end)
			loss = (start_loss + end_loss)
			ret_outputs = (loss,) + ret_outputs
		
		return ret_outputs # (loss), start_logits, end_logits
	
	def freeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = False
	
	def unfreeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = True

# ============================================================================

# Model & Optimizer setting
model = BertForExtraction.from_pretrained(output_dir)
model = model.cuda()
device = torch.device("cuda")

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(infer_data)))

# Put model in evaluation mode
model.eval()

# Save the labeled data
output_dir = './saved/bert-extract/'

def get_selected_text(text, start_idx, end_idx, offsets):
	selected_text = ""
	for ix in range(start_idx, end_idx + 1):
		selected_text += text[offsets[ix][0]: offsets[ix][1]]
		if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
			selected_text += " "
	return selected_text

# Model Inference
selected_texts = []
for batch in infer_dataloader:	
	batch = tuple(t.to(device) for t in batch)

	b_input_ids, b_input_mask, b_offsets, b_index  = batch

	with torch.no_grad():
		outputs = model(b_input_ids, attention_mask=b_input_mask)

	start_logits, end_logits = outputs
	start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
	end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

	b_offsets = b_offsets.cpu().detach().numpy()
	b_index = b_index.cpu().detach().numpy()

	for i in range(len(b_input_ids)):
		start_pred = np.argmax(start_logits[i])
		end_pred = np.argmax(end_logits[i])
		if start_pred > end_pred:
			pred = infer_set[b_index[i]]
		else:
			pred = get_selected_text(infer_set[b_index[i]], start_pred, end_pred, b_offsets[i])
		selected_texts.append(pred)

# Output the extracted results
output_file_path = r"./data/train_full_ex.json"

for idx, sample in enumerate(infer_list):
	sample["extracted"] = selected_texts[idx]

with open(output_file_path, 'w', encoding="utf-8") as f:
	json.dump(infer_list, f, ensure_ascii=False, indent=0, separators=(',', ':'))