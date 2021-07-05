from typing import Dict, Optional

from overrides import overrides
import torch
import torch.nn as nn

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.attention.additive_attention import AdditiveAttention
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


class BasicClassifier(Model):
    """
    This `Model` integrates CNN, RNN, RCNN and Transformer classifier. 

    # Parameters
    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2vec_encoder : `Encoder`
        Required encoder layer, either a seq2seq or seq2vector. This encoder will pool its output. 
    feedforward : `FeedForward`, optional, (default = `None`)
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels : `int`, optional (default = `None`)
        Number of labels to project to in classification layer. By default, the classification layer will
        project to the size of the vocabulary namespace corresponding to labels.
    label_namespace : `str`, optional (default = `"labels"`)
        Vocabulary namespace corresponding to labels. By default, we use the "labels" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    modelname : `str`, optional (default = `None`)
        The name of the deep-learning model
    conv_layer : `str`, optional (default = `None`)
        If provided, an extra convolutional layer will be activated
    mode : `str` optional (default: `None`)
        if it is in `train` mode, the output only contains the accuracy, 
        if is is in `eval` mode, the ourput will have accuracy and f1 score 
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2vec_encoder,
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        modelname: str = None,
        conv_layer=None,
        mode=None,
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        self._modelname = modelname
        self._mode = mode

        # additional parameters only for Transformer
        if modelname == "TransformerClassifier":
            # transformer output -> self attention 
            input_dim = self._text_field_embedder.get_output_dim()
            output_dim = feedforward.get_input_dim()
            self._transformforward = nn.Linear(input_dim, output_dim)
            self._selfattn = AdditiveAttention(vector_dim=input_dim, matrix_dim=input_dim, normalize=True)

        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace
        self._namespace = namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)
        self._conv_layer = conv_layer
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._f1 = F1Measure(positive_label=1)
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`
        # Returns
        An output dictionary consisting of:
            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        embedded_text_seq = self._seq2vec_encoder(embedded_text, mask=mask)

        # special operations for RCNN
        if self._modelname == 'RCNNClassifier':
            # an extra CNN layer + global maxpooling
            embedded_text_concat = torch.cat((embedded_text_seq, embedded_text), dim=2).transpose(1, 2)
            new_embedded = torch.tanh(self._conv_layer(embedded_text_concat))
            embedded_text_seq, input_indices = torch.max(new_embedded, 2)	

        if self._dropout:
            embedded_text_seq = self._dropout(embedded_text_seq)
        
        # special operations for Transformer
        if self._modelname == "TransformerClassifier":
            # mean-pooling the encoder output and go through the self-attention layer
            context = torch.mean(embedded_text_seq, axis=1)
            attn_weights = self._selfattn(context, embedded_text_seq)
            attn_context = util.weighted_sum(embedded_text_seq, attn_weights)
            embedded_text_seq = self._transformforward(attn_context)

        if self._feedforward is not None:
            embedded_text_seq = self._feedforward(embedded_text_seq)

        logits = self._classification_layer(embedded_text_seq)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            self._f1(logits, label)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        if self._mode == 'train':
            metrics = {"accuracy": self._accuracy.get_metric(reset)}
        elif self._mode == 'eval':
            metrics = {"accuracy": self._accuracy.get_metric(reset),
            "f1-score": self._f1.get_metric(reset)}
        else:
            metrics = None
            print('mode should be either "train" or "eval"')
        return metrics
