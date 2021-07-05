from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


class HAN(Model):
    """
        word_embedder(text)
        word_emb : [B, S, L, E] # [B, L, E]

        word_encoder LSTM(hidden_size)

        word_reshaped = word_emb.reshape (BxS, L, E)
        word_hidden, word_output = LSTM(word_reshaped) hidden (BXS, L, E) output (BXS, L, E)
        u_w = word_output[-1]

        attn = attn_function(k=u_w, q=word_hidden, v=word_hidden) [BXS, L]
        0.1 0.6 0.3

        uw_new = weighted_sum(attn, word_output)

        word_output [B, S, E]
        attn = allennlp

        sent_encoder LSTM(hidden_size)
    """
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,

        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        namespace: str = "tokens",
        initializer: InitializerApplicator = InitializerApplicator(),
        # --------------------------------------------------------------------------
        word_encoder_embedding_size: int = 200, # according to word to vector methods
        word_gru_hidden: int = 100,
        sent_gru_hidden: int = 100,
        bidirectional: bool = True,
        #----------------------------------------------------------------------------
        **kwargs,
    ) -> None:
        
        super().__init__(vocab, **kwargs)
        self.bidirectional = bidirectional
        self._text_field_embedder = text_field_embedder

        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            raise NotImplementedError()

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
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)

        #---------------------------------------------------------
        self.word_encoder = torch.nn.GRU(word_encoder_embedding_size, word_gru_hidden, bidirectional = self.bidirectional, batch_first=True)

        from allennlp.modules.attention.additive_attention import AdditiveAttention
        if self.bidirectional == True:
            self.word_attention = AdditiveAttention(vector_dim=word_gru_hidden*2, matrix_dim=word_gru_hidden*2, normalize = True)
            self.sent_attention = AdditiveAttention(vector_dim=sent_gru_hidden*2, matrix_dim=sent_gru_hidden*2, normalize = True)
            self.sentence_encoder = torch.nn.GRU(word_gru_hidden * 2, sent_gru_hidden, bidirectional = self.bidirectional, batch_first=True)
        else:
            self.word_attention = AdditiveAttention(vector_dim=word_gru_hidden, matrix_dim=word_gru_hidden, normalize = True)
            self.sent_attention = AdditiveAttention(vector_dim=sent_gru_hidden, matrix_dim=sent_gru_hidden, normalize = True)
            self.sentence_encoder = torch.nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional = self.bidirectional, batch_first=True)
        #---------------------------------------------------------

        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        #---------------------------------------------------------------
        embedded_text = self._text_field_embedder(tokens)

        # masking: we need to reshape the tokens to get the right mask
        tokens_content = tokens['tokens']['tokens']

        tokens_reshape = tokens_content.reshape([tokens_content.shape[0] * tokens_content.shape[1], tokens_content.shape[-1]])
        word_mask = get_text_field_mask({'tokens':{'tokens':tokens_reshape}})

        size_B = embedded_text.shape[0]
        size_S = embedded_text.shape[1]

        embedded_text_BS_L_E = embedded_text.reshape([embedded_text.shape[0] * embedded_text.shape[1], 
                                                        embedded_text.shape[2], 
                                                        embedded_text.shape[3]])
        
        # masking 
        embedded_text_BS_L_E_masked = embedded_text_BS_L_E.reshape([-1, embedded_text_BS_L_E.shape[-1]]).transpose(0,1).masked_fill(~word_mask.view(-1), 0.0)
        embedded_text_BS_L_E_masked = embedded_text_BS_L_E_masked.transpose(0,1).reshape(embedded_text_BS_L_E.size())

        output_word, state_word = self.word_encoder(embedded_text_BS_L_E_masked) 

        # make it bidirectional
        if self.bidirectional == True:
            state_word = state_word.transpose(0,1)
            state_word = state_word.reshape([state_word.shape[0], 2 * state_word.shape[-1]])
        else:
            state_word = state_word.reshape([state_word.shape[1], state_word.shape[-1]])
        word_attention = self.word_attention(state_word, output_word)
        
        # batch size
        output_word_weight = util.weighted_sum(output_word, word_attention)
        
        # do masking
        sent_mask = get_text_field_mask(tokens) # get sentence mask, if we donot reshape it, it will has 3 dimension and the get_text_field function will regard it as [batch, token(sentence), features]
        output_word_weight_masked = output_word_weight.reshape([-1, output_word_weight.shape[-1]]).transpose(0,1).masked_fill(~sent_mask.view(-1), 0.0)
        output_word_weight_masked = output_word_weight_masked.transpose(0,1).reshape(output_word_weight.size())

        output_word_weight_B_S_E = output_word_weight_masked.reshape([size_B, size_S, -1])
        output_sent, state_sent = self.sentence_encoder(output_word_weight_B_S_E)

        if self.bidirectional == True:
            state_sent = state_sent.transpose(0,1)
            state_sent = state_sent.reshape([state_sent.shape[0], 2 * state_sent.shape[-1]])
        else:
            state_sent = state_sent.reshape([state_sent.shape[1], state_sent.shape[2]])
        
        sent_attention = self.sent_attention(state_sent, output_sent)
        output_sent_weight = util.weighted_sum(output_sent, sent_attention)
        # 16 x 100
        #-------------------------------------------------------------------------------------

        if self._feedforward is not None:
            embedded_text = self._feedforward(output_sent_weight)

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
