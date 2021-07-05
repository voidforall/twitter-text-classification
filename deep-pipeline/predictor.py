from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField

class TextClassifierPredictor(Predictor):

	def predict(self, text: str) -> JsonDict:
		return self.predict_json({"text": text})

	@overrides
	def _json_to_instance(self, json_dict: JsonDict) -> Instance:
		"""
		Expects JSON that looks like `{"text": "..."}`.
		Runs the underlying model, and adds the `"label"` to the output.
		"""
		text = json_dict["text"]
		return self._dataset_reader.text_to_instance(text)

	@overrides
	def predictions_to_labeled_instances(
		self, instance: Instance, outputs: Dict[str, numpy.ndarray]
	) -> List[Instance]:
		new_instance = instance.duplicate()
		label = numpy.argmax(outputs["probs"])
		new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
		return [new_instance]
