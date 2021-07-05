import os
import json
import time
import datetime
import torch
import numpy as np
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

# Load train data / valid data
train_datapath = r"./data/train_full_ex.json"
valid_datapath = r"./data/valid_full_ex.json"

with open(train_datapath, "r", encoding="utf-8") as f:
	train_list = json.load(f)
with open(valid_datapath, "r", encoding="utf-8") as f:
	valid_list = json.load(f)

train_set = [(sample["text"].strip(), sample["label"], sample["extracted"].strip()) for sample in train_list]
valid_set = [(sample["text"].strip(), sample["label"], sample["extracted"].strip()) for sample in valid_list]

# Tokenization
# load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(r"bert-base-uncased")

# Apply BERT tokenizer for tokenization
max_len = 50
train_inputs_text, train_masks_text = dataprepare.bert_tokenize(train_set, tokenizer, max_len)
train_inputs_ex, train_masks_ex = dataprepare.bert_tokenize(train_set, tokenizer, max_len, is_ex=True)
valid_inputs_text, valid_masks_text = dataprepare.bert_tokenize(valid_set, tokenizer, max_len)
valid_inputs_ex, valid_masks_ex = dataprepare.bert_tokenize(valid_set, tokenizer, max_len, is_ex=True)

train_labels = torch.tensor([i[1] for i in train_set])
valid_labels = torch.tensor([i[1] for i in valid_set])

# DataLoader is really important in saving memory while training
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 64 	# author recommend batch size of 16 or 32 for fine-tuning BERT
# Create the DataLoader
train_data = TensorDataset(train_inputs_text, train_masks_text, train_inputs_ex, train_masks_ex, train_labels)
train_dataloader = DataLoader(
		train_data, 
		sampler=RandomSampler(train_data), 
		batch_size=batch_size
	)

validation_data = TensorDataset(valid_inputs_text, valid_masks_text, valid_inputs_ex, valid_masks_ex, valid_labels)
validation_dataloader = DataLoader(
		validation_data, 
		sampler=SequentialSampler(validation_data), # maintain the sequential sampling for validation
		batch_size=batch_size
	)

# Declare the model structure
class BertForSentiment(BertPreTrainedModel):
	"""
		This module is composed of BERT-base-uncased with multi-layer
		Siamense structure on top of the pooled output, specific
		for downstream task - sentiment classification (binary)
	"""
	def __init__(self, config):
		super().__init__(config)

		self.bert = BertModel(config)
		self.dropout_text = nn.Dropout(0.2)
		self.dropout_ex   = nn.Dropout(0.2)
		self.transform_text_layer = nn.Sequential(nn.Linear(config.hidden_size, 64),
										nn.LayerNorm(64),
										nn.ReLU(),
										nn.Dropout(0.2),
										)
		self.transform_ex_layer = nn.Sequential(nn.Linear(config.hidden_size, 64),
										nn.LayerNorm(64),
										nn.ReLU(),
										nn.Dropout(0.2),
										)	
		self.classifier = nn.Linear(64*2, 1)
		self.sigmoid = nn.Sigmoid()

		self.init_weights()
	
	def forward(
		self,
		input_ids_text=None,
		attention_mask_text=None,
		input_ids_ex=None,
		attention_mask_ex=None,
		labels=None,
	):
	
		outputs_text = self.bert(
			input_ids_text,
			attention_mask=attention_mask_text
		)
		outputs_ex = self.bert(
			input_ids_ex,
			attention_mask=attention_mask_ex
		)
		
		pooled_output_text = outputs_text[1]
		pooled_output_text = self.dropout_text(pooled_output_text) # shape: (batch_num, hidden_size)
		pooled_output_ex   = outputs_ex[1]
		pooled_output_ex   = self.dropout_ex(pooled_output_ex) # shape: (batch_num, hidden_size)

		transform_text_emb = self.transform_text_layer(pooled_output_text)
		transform_ex_emb   = self.transform_ex_layer(pooled_output_ex)

		logits = self.sigmoid(self.classifier(torch.cat((transform_text_emb, transform_ex_emb), dim=-1))) # shape: (batch_num, num_labels)
		
		outputs = (logits,)
		
		if labels is not None:
			loss_fct = torch.nn.BCELoss()
			labels = labels.float()
			loss = loss_fct(logits.view(-1), labels.view(-1))
			outputs = (loss,) + outputs
			
		return outputs # (loss), logits
	
	def freeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = False
	
	def unfreeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = True


# Model & Optimizer setting
model = BertForSentiment.from_pretrained("bert-base-uncased")
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

# metric: accuracy
def accuracy(preds, labels):
	"""
	   Function to calculate the accuracy of our predictions vs labels.
	"""
	pred_round = np.round(preds)
	labels_reshaped = labels[:, np.newaxis]
	acc = np.sum(pred_round == labels_reshaped) / labels.shape[0] 
	return acc


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
log_accs = []

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
	eval_accuracy = 0
	nb_eval_steps = 0

	model.train()

	for step, batch in enumerate(train_dataloader):

		if step % 100 == 0 and not step == 0:
			elapsed = format_time(time.time() - t0)
			print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. Loss: {}  Acc: {}'.format(step, len(train_dataloader), elapsed, total_loss/nb_eval_steps, eval_accuracy/nb_eval_steps))

		b_input_ids_text = batch[0].to(device)
		b_input_mask_text = batch[1].to(device)
		b_input_ids_ex = batch[2].to(device)
		b_input_mask_ex = batch[3].to(device)
		b_labels = batch[4].to(device)

		model.zero_grad()
		outputs = model(input_ids_text=b_input_ids_text,
						attention_mask_text=b_input_mask_text,
						input_ids_ex=b_input_ids_ex,
						attention_mask_ex=b_input_mask_ex,
						labels=b_labels
						)

		loss = outputs[0]
		total_loss += loss.item()
		batch_losses.append(loss.item())
		loss.backward()

		logits = outputs[1]
		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()

		torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
		optimizer.step()
		scheduler.step()

		eval_accuracy += accuracy(logits, label_ids)
		nb_eval_steps += 1

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
	eval_accuracy = 0
	nb_eval_steps = 0

	# Evaluate data for one epoch
	for batch in validation_dataloader:
		
		batch = tuple(t.to(device) for t in batch)
		b_input_ids_text, b_input_mask_text, b_input_ids_ex, b_input_mask_ex, b_labels = batch
		
		with torch.no_grad():

			outputs = model(input_ids_text=b_input_ids_text,
							attention_mask_text=b_input_mask_text,
							input_ids_ex=b_input_ids_ex,
							attention_mask_ex=b_input_mask_ex
							)
		
		logits = outputs[0]

		logits = logits.detach().cpu().numpy()
		label_ids = b_labels.to('cpu').numpy()
		
		eval_accuracy += accuracy(logits, label_ids)
		nb_eval_steps += 1

	log_accs.append(eval_accuracy/nb_eval_steps)

	print("  Accuracy: {0:.5f}".format(eval_accuracy/nb_eval_steps))
	print("  Validation took: {:}".format(format_time(time.time() - t0)))

# Save the model's parameters
output_dir = './saved/bert_full_exsiamense/'

# Create output directory if needed
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

# Save the model's parameters
print("Saving model to %s" % output_dir)
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir) # can be then directly loaded with load_pretrained()
tokenizer.save_pretrained(output_dir)

# Save the model's logs
log_data = {
	"train_batch_losses": batch_losses,
	"train_epoch_losses": loss_values,
	"valid_epoch_accs": log_accs
}

with open(output_dir + "log.json", 'w', encoding="utf-8") as f:
	json.dump(log_data, f, ensure_ascii=False, indent=0, separators=(',', ':'))