import sys, os
import json

"""
This file is used to convert the .txt file for prediction to .json, the input file type for our model training and evaluation.
"""

if __name__ == '__main__':
	test_data_path = r"../data/test_data_clean.txt"

	# load the data to list
	with open(test_data_path, "r", encoding="utf-8") as data_file:
		test_data = data_file.readlines()

	# turn to the json format and save it 
	test_json = [{"text": x} for x in test_data]

	# store the splitted dataset as json file
	with open(r'../data/predict.json', 'w', encoding="utf-8") as f:
		json.dump(test_json, f, ensure_ascii=False, indent=0, separators=(',', ':'))
