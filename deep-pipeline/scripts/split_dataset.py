import sys, os
import json
from sklearn.model_selection import train_test_split
import argparse

def split_dataset(X, y, seed, test_size=0.05):
	""" Process and split the dataset (train/valid/test) """
	train_X, dev_test_X, train_y, dev_test_y = train_test_split(X, y, test_size=test_size*2, random_state=seed)
	dev_X, test_X, dev_y, test_y = train_test_split(dev_test_X, dev_test_y, test_size=0.5, random_state=seed)
	return train_X, train_y, dev_X, dev_y, test_X, test_y

if __name__ == '__main__':
	# arguments
	parser = argparse.ArgumentParser(description="data splitting")
	parser.add_argument('--dataset', dest = "data",type=str,default=None, help="process which dataset: subset/fullset", required=True)
	args = parser.parse_args()

	# declare the data path
	if args.data == 'subset':
		pos_data_path = r"../data/train_pos_clean.txt"
		neg_data_path = r"../data/train_neg_clean.txt"
		POSTFIX = r"_subset"	
	elif args.data == 'fullset':
		pos_data_path = r"../data/train_pos_full_clean.txt"
		neg_data_path = r"../data/train_neg_full_clean.txt"
		POSTFIX = '_full'
	else:
		print('no such dataset')
		
	# load the data to list
	with open(pos_data_path, "r", encoding="utf-8") as data_file_pos:
		pos_X = data_file_pos.readlines()
	pos_y = [1 for i in range(len(pos_X))]
	with open(neg_data_path, "r", encoding="utf-8") as data_file_neg:
		neg_X = data_file_neg.readlines()
	neg_y = [0 for i in range(len(neg_X))]
	dataset_X = pos_X + neg_X
	dataset_y = pos_y + neg_y

	# split the dataset (TRAIN/VALID/TEST, where VALID_SIZE = TEST_SIZE)
	TEST_SIZE = 0.05
	seed = 42
	train_X, train_y, dev_X, dev_y, test_X, test_y = split_dataset(dataset_X, dataset_y, seed, TEST_SIZE)
	train = [{"text": x, "label": y} for x, y in zip(train_X, train_y)]
	valid = [{"text": x, "label": y} for x, y in zip(dev_X, dev_y)]
	test = [{"text": x, "label": y} for x, y in zip(test_X, test_y)]

	# store the splitted dataset as json file
	with open(r'../data/train' + POSTFIX + '.json', 'w', encoding="utf-8") as f:
		json.dump(train, f, ensure_ascii=False, indent=0, separators=(',', ':'))
	with open(r'../data/valid'+ POSTFIX + '.json', 'w', encoding="utf-8") as f:
		json.dump(valid, f, ensure_ascii=False, indent=0, separators=(',', ':'))
	with open(r'../data/test'+ POSTFIX + '.json', 'w', encoding="utf-8") as f:
		json.dump(test, f, ensure_ascii=False, indent=0, separators=(',', ':'))


