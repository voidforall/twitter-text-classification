import numpy as np
import json

# Load the ground truth labels
# test_file_path = r"./data/test_subset.json"
test_file_path = r"./data/test_full.json"
with open(test_file_path, "r", encoding="utf-8") as f:
	test_list = json.load(f)
test_labels = np.array([sample["label"] for sample in test_list])

# Apply vote ensemble of the candidate model results
# result_path = ["./saved/bert2_subset/", "./saved/bert_subset/", "./saved/bert_subset_ex/", 
# 				"./saved/bert_subset_exsiamense/", "./saved/roberta/"]
result_path = ["./saved/bert2_full/", "./saved/bert_full/", "./saved/bert_full_ex/", 
				"./saved/bert_full_exsiamense/", "./saved/roberta_full/"]

votes_test = np.zeros(len(test_labels))
votes_predict = np.zeros(10000)
for i in result_path:
	temp_test = np.load(i + "full_test.npy")
	votes_test = votes_test + temp_test
	temp_predict = np.load(i + "full_predict.npy")
	votes_predict = votes_predict + temp_predict

predictions = votes_test > 2.5
predictions = [1 if p==True else 0 for p in predictions]
# Evaluate the performance on test set
from sklearn.metrics import accuracy_score, f1_score

acc = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
print('Total acc: %.5f' % acc)
print('Total f1: %.5f' % f1)


# Output the final submission file 
from utils.result_helpers import create_csv_submission
output_file_path = r"./submission.csv"
predictions = votes_predict > 0
predictions = [1 if p==True else -1 for p in predictions]
ids = list(range(1, len(votes_predict) + 1))
create_csv_submission(ids, predictions, output_file_path)
