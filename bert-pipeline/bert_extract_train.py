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


print("BERT Training starts!")

# Fix random seeds and specify GPU device
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# Load extraction dataset
data_path = r"./extract.csv"
data_df = pd.read_csv(data_path)
data_df = data_df[data_df.sentiment != "neutral"] # remove neutral tweets
data_df.dropna(axis = 0, how ='any',inplace=True) # drop na rows

# Preprocess tweet according to Glove tokenize
preprocessor = dataprepare.GloVePreprocessor(True)
data_df["text"] = data_df["text"].apply(lambda x: preprocessor.preprocess(x))
data_df["selected_text"] = data_df["selected_text"].apply(lambda x: preprocessor.preprocess(x))

# Use Bert tokenizer to re-tokenize (WordPiece)
tokenizer = tokenizers.BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
tokenizer.add_special_tokens(["<allcaps>", "<elong>", "<url>", "<smile>", "<lolface>", "<sadface>", "<neutralface>", "<heart>", "<number>", "<repeat>"])

# Extract the required fields of data (input_ids, masks, targets_start, targets_end, sentiment)
max_len = 200
dataset = []
for i in range(len(data_df)):
	output_dict = dataprepare.process_extract_csv(data_df.iloc[i]["text"], data_df.iloc[i]["selected_text"], data_df.iloc[i]["sentiment"], tokenizer, max_len)
	if output_dict == None:
		continue
	dataset.append((output_dict["ids"], output_dict["mask"], output_dict["targets_start"], 
		output_dict["targets_end"], output_dict["sentiment"], output_dict["tweet_offsets"], output_dict["tweet_text"]))

# Split the dataset
from sklearn.model_selection import train_test_split
train_dataset, valid_dataset, _, _ = train_test_split(dataset, [0]*len(dataset), test_size=0.05, random_state=42)
train_inputs, train_masks = torch.tensor([i[0] for i in train_dataset]), torch.tensor([i[1] for i in train_dataset])
valid_inputs, valid_masks = torch.tensor([i[0] for i in valid_dataset]), torch.tensor([i[1] for i in valid_dataset])

train_start, train_end = torch.tensor([i[2] for i in train_dataset]), torch.tensor([i[3] for i in train_dataset])
valid_start, valid_end = torch.tensor([i[2] for i in valid_dataset]), torch.tensor([i[3] for i in valid_dataset])
train_labels = torch.tensor([i[4] for i in train_dataset])
valid_labels = torch.tensor([i[4] for i in valid_dataset])
valid_index = torch.tensor([i for i in range(len(valid_dataset))])

# DataLoader is really important in saving memory while training
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 32     # author recommend batch size of 16 or 32 for fine-tuning BERT
# Create the DataLoader
train_data = TensorDataset(train_inputs, train_masks, train_start, train_end, train_labels)
train_dataloader = DataLoader(
		train_data, 
		sampler=RandomSampler(train_data), 
		batch_size=batch_size
	)

validation_data = TensorDataset(valid_inputs, valid_masks, valid_start, valid_end, valid_labels, valid_index)
validation_dataloader = DataLoader(
		validation_data, 
		sampler=SequentialSampler(validation_data), # maintain the sequential sampling for validation
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


# Model & Optimizer setting
model = BertForExtraction.from_pretrained("bert-base-uncased")
model.resize_token_embeddings(tokenizer.get_vocab_size())
model = model.cuda()

optimizer = AdamW(
		model.parameters(),
		lr = 3e-5,
		eps = 1e-8
	)

# Number of training epochs, authors recommend fine-tuning epochs between 2 and 4
epochs = 2

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
											num_warmup_steps = 0, # Default value in run_glue.py
											num_training_steps = total_steps)

# metric: jaccard score
# Ref: https://www.kaggle.com/shoheiazuma/tweet-sentiment-roberta-pytorch
def get_selected_text(text, start_idx, end_idx, offsets):
	selected_text = ""
	for ix in range(start_idx, end_idx + 1):
		selected_text += text[offsets[ix][0]: offsets[ix][1]]
		if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
			selected_text += " "
	return selected_text

def jaccard(str1, str2): 
	a = set(str1.lower().split()) 
	b = set(str2.lower().split())
	c = a.intersection(b)
	return float(len(c)) / (len(a) + len(b) - len(c))

def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
	start_pred = np.argmax(start_logits)
	end_pred = np.argmax(end_logits)
	if start_pred > end_pred:
		pred = text
	else:
		pred = get_selected_text(text, start_pred, end_pred, offsets)
		
	true = get_selected_text(text, start_idx, end_idx, offsets)
	
	return jaccard(true, pred)


def format_time(elapsed):
	'''
		Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))


# Store the average loss after each epoch so we can plot them.
batch_losses = []
loss_values = []
log_jaccards = []

device = torch.device("cuda")

# Train & Validation
for epoch_i in range(0, epochs):
	
	# ========================================
	#               Training
	# ========================================
	print("")
	print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
	print('Training...')
	t0 = time.time()

	total_loss = 0

	model.train()

	for step, batch in enumerate(train_dataloader):

		if step % 100 == 0 and not step == 0:
			elapsed = format_time(time.time() - t0)
			print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

		b_input_ids = batch[0].to(device)
		b_input_mask = batch[1].to(device)
		b_start = batch[2].to(device)
		b_end = batch[3].to(device)
		b_labels = batch[4].to(device)

		model.zero_grad()
		outputs = model(
			input_ids=b_input_ids, 
			token_type_ids=None, 
			attention_mask=b_input_mask,
			labels=b_labels,
			start=b_start,
			end=b_end
		)

		loss = outputs[0]

		total_loss += loss.item()
		batch_losses.append(loss.item())
		loss.backward()

		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		scheduler.step()

	avg_train_loss = total_loss / len(train_dataloader)
	loss_values.append(avg_train_loss)

	print("")
	print("  Average training loss: {0:.2f}".format(avg_train_loss))
	print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
		
	# ========================================
	#               Validation
	# ========================================
	print("")
	print("Running Validation...")
	t0 = time.time()

	model.eval()

	# Tracking variables 
	eval_jaccard = 0
	nb_eval_steps = 0

	# Evaluate data for one epoch
	for batch in validation_dataloader:
		
		batch = tuple(t.to(device) for t in batch)
		b_input_ids, b_input_mask, b_start, b_end, b_labels, b_index = batch
		
		with torch.no_grad():
			outputs = model(b_input_ids, 
							token_type_ids=None, 
							attention_mask=b_input_mask)

		b_start = b_start.cpu().detach().numpy()
		b_end = b_end.cpu().detach().numpy()
		start_logits, end_logits = outputs
		start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
		end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
		
		b_index = b_index.cpu().detach().numpy()

		for i in range(len(b_input_ids)):
			jaccard_score = compute_jaccard_score(
				valid_dataset[int(b_index[i])][6],
				b_start[i],
				b_end[i],
				start_logits[i], 
				end_logits[i], 
				valid_dataset[int(b_index[i])][5]
			)
			eval_jaccard += jaccard_score

		nb_eval_steps += 1

	log_jaccards.append(eval_jaccard/nb_eval_steps/batch_size)

	print("  Jaccards: {0:.5f}".format(eval_jaccard/nb_eval_steps/batch_size))
	print("  Validation took: {:}".format(format_time(time.time() - t0)))

# Save the model's parameters
output_dir = './saved/bert-extract/'

# Create output directory if needed
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Save the model's parameters
print("Saving model to %s" % output_dir)
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir) # can be then directly loaded with load_pretrained()
tokenizer.save(output_dir)

# Save the model's logs
log_data = {
	"train_batch_losses": batch_losses,
	"train_epoch_losses": loss_values,
	"valid_epoch_jaccards": log_jaccards
}

with open(output_dir + "log.json", 'w', encoding="utf-8") as f:
	json.dump(log_data, f, ensure_ascii=False, indent=0, separators=(',', ':'))