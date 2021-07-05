import os
import json
import torch
import numpy as np
import random
from transformers import *
import torch.nn as nn
import dataprepare


print("BERT Evaluation starts!")

# fix random seed and specify GPU device
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Load test data / predict data
test_datapath = r"./data/test_full_ex.json"
predict_datapath = r"./data/test_data_ex.json"

with open(test_datapath, "r", encoding="utf-8") as f:
	test_list = json.load(f)
with open(predict_datapath, "r", encoding="utf-8") as f:
	predict_list = json.load(f)

test_set = [(sample["text"].strip(), sample["label"], sample["extracted"].strip()) for sample in test_list]
predict_set = [(sample["text"].strip(), 0, sample["extracted"].strip()) for sample in predict_list] # fake labels in predict set

output_dir = r"./saved/bert_full_ex"

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(output_dir)

# Apply BERT tokenizer for tokenization
max_len = 200
test_inputs, test_masks, test_types = dataprepare.bert_tokenize_ex(test_set, tokenizer, max_len)
predict_inputs, predict_masks, predict_types = dataprepare.bert_tokenize_ex(predict_set, tokenizer, max_len)
test_labels = torch.tensor([i[1] for i in test_set])
predict_labels = torch.tensor([i[1] for i in predict_set])

# DataLoader is really important in saving memory while training
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
batch_size = 32 	# author recommend batch size of 16 or 32 for fine-tuning BERT
# Create the DataLoader
test_data = TensorDataset(test_inputs, test_masks, test_types, test_labels)
test_dataloader = DataLoader(
		test_data, 
		sampler=SequentialSampler(test_data), 
		batch_size=batch_size
	)

predict_data = TensorDataset(predict_inputs, predict_masks, predict_types, predict_labels)
predict_dataloader = DataLoader(
		predict_data, 
		sampler=SequentialSampler(predict_data),
		batch_size=batch_size
	)

# Declare the model structure
class BertForSentiment(BertPreTrainedModel):
	"""
		This module is composed of BERT-base-uncased with a
		linear layer on top of the pooled output, specific
		for downstream task - sentiment classification (binary)
		
		Paired sequences is applied, with `token_type_ids` to specify
		the sequence id. Two sequences are respectively the original
		tweet and the extracted sentiment phrase.	
	"""
	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config)
		self.dropout = nn.Dropout(0.3)
		self.classifier = nn.Linear(config.hidden_size, 1)
		self.sigmoid = nn.Sigmoid()

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
	):
	
		outputs = self.bert(
			input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
		)
		
		pooled_output = outputs[1]
		pooled_output = self.dropout(pooled_output) # shape: (batch_num, hidden_size)
		logits = self.sigmoid(self.classifier(pooled_output)) # shape: (batch_num, num_labels)
		
		outputs = (logits,) + outputs[2:]
		
		if labels is not None:
			loss_fct = torch.nn.BCELoss()
			labels = labels.float()
			loss = loss_fct(logits.view(-1), labels.view(-1))
			outputs = (loss,) + outputs
			
		return outputs # (loss), logits, (hidden_states), (attentions) 
	
	def freeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = False
	
	def unfreeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = True


# ============================================================================

# model setting
model = BertForSentiment.from_pretrained(output_dir)
model = model.cuda()

# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(test_data)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []
device = torch.device("cuda")

# Predict 
for batch in test_dataloader:	
	batch = tuple(t.to(device) for t in batch)

	b_input_ids, b_input_mask, b_input_type, b_labels = batch

	with torch.no_grad():
		outputs = model(b_input_ids, token_type_ids=b_input_type, 
						attention_mask=b_input_mask)

	logits = outputs[0]

	logits = logits.detach().cpu().numpy()
	label_ids = b_labels.to('cpu').numpy()
	
	predictions.append(logits)
	true_labels.append(label_ids)
print('Evaluation on TEST set has DONE.')

# Calculate scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print('Calculating F1 and acc for test set.')

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = np.round(flat_predictions).flatten()
flat_true_labels = np.concatenate(true_labels, axis=0)

acc = accuracy_score(flat_true_labels, flat_predictions)
f1 = f1_score(flat_true_labels, flat_predictions)

print('Total acc: %.5f' % acc)
print('Total f1: %.5f' % f1)

np.save(output_dir + "/full_test", flat_predictions)


#  ===================================================
# output predictions on predict file
predictions = []
for batch in predict_dataloader:	
	batch = tuple(t.to(device) for t in batch)

	b_input_ids, b_input_mask, b_input_type, b_labels = batch

	with torch.no_grad():
		outputs = model(b_input_ids, token_type_ids=b_input_type, 
						attention_mask=b_input_mask)

	logits = outputs[0]

	logits = logits.detach().cpu().numpy()
	
	predictions.append(logits)

flat_predictions = np.concatenate(predictions, axis=0)
flat_predictions = list(np.round(flat_predictions).flatten())
ids = list(range(1, len(flat_predictions) + 1))
flat_predictions = [p if p==1 else -1 for p in flat_predictions]

from utils.result_helpers import create_csv_submission
output_file_path = r"./saved/bert_subset_ex/submission.csv"
create_csv_submission(ids, flat_predictions, output_file_path)

print("[STATUS] Evaluation ends, all results have been written into %s" % output_file_path)


np.save(output_dir + "/full_predict", flat_predictions)
