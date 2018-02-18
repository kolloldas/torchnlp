from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common.hparams import HParams
from .common.model import Model, gen_model_dir
from .modules import transformer, outputs

import os

VOCABS_FILE = 'vocabs.pt'

class Tagger(Model):
    """
    Abstract base class that adds the following boilerplate for
    sequence tagging tasks:
    - Word Embeddings
    - Character Embeddings
    - Tag projection
    - CRF
    Derived classes implement the compute() method and not forward(). 
    This is so that projection and other layers can be added
    """
    def __init__(self, hparams=None, vocabs=None):
        """
        Parameters:
            hparams: Instance of HParams class
            num_tags: Number of output tags
            vocabs: tuple of (word vocab, char vocab, tags vocab). Each is an
                    instance of torchtext.vocab.Vocab.
            NOTE: If word_vocab.vectors is available it will initialize the embeddings
            and with word_vocab.vectors make it non-trainable
        """
        super(Tagger, self).__init__(hparams)

        if vocabs is None or not isinstance(vocabs, tuple) or len(vocabs) != 3:
            raise ValueError('Must provide vocabs 3-tuple')

        vocab_word, vocab_char, vocab_tags = vocabs
        if vocab_word is None:
            raise ValueError('Must provide vocab_word')
        if vocab_tags is None:
            raise ValueError('Must provide vocab_word')

        self.vocabs = vocabs
        self.vocab_tags = vocab_tags # Needed during eval and prediction
        self.embedding_word = nn.Embedding(len(vocab_word), hparams.embedding_size_word)
        self.embedding_char = None

        if vocab_char is not None and hparams.embedding_size_char > 0:
            self.embedding_char = nn.Embedding(len(vocab_char), hparams.embedding_size_char)
        
        if vocab_word.vectors is not None:
            if hparams.embedding_size_word != vocab_word.vectors.shape[1]:
                raise ValueError('embedding_size should be {} but got {}'
                                .format(vocab_word.vectors.shape[1], 
                                        hparams.embedding_size_word))
            self.embedding_word.weight.data.copy_(vocab_word.vectors)
            self.embedding_word.weight.requires_grad = False
        
        # self.tags_projection = nn.Linear(hparams.hidden_size, len(vocab_tags))
        # self.loss_func = nn.NLLLoss()

        if hparams.use_crf:
            self.output_layer = outputs.CRFOutputLayer(hparams.hidden_size, len(vocab_tags))
        else:
            self.output_layer = outputs.SoftmaxOutputLayer(hparams.hidden_size, len(vocab_tags))

    def _embed_compute(self, batch):
        inputs_word_emb = self.embedding_word(batch.inputs_word)
        inputs_char_emb = None
        if self.embedding_char is not None:
            inputs_char_emb = self.embedding_char(batch.inputs_char.view(-1, 
                                                  batch.inputs_char.shape[-1]))

        return self.compute(inputs_word_emb, inputs_char_emb)
        # tag_logits = self.tags_projection(outputs)

        # return tag_logits

    def forward(self, batch):
        """
        NOTE: batch must have the following attributes:
            inputs_word, inputs_char, labels
        """
        hidden = self._embed_compute(batch)
        return self.output_layer(hidden)
        # probs = F.softmax(tag_logits, -1)

        # #TODO: Add CRF layer
        # #TODO: Add beam search somewhere :)
        # _, predictions = torch.max(probs, dim=2)

        # return predictions
        

    def loss(self, batch, compute_predictions=False):
        """
        NOTE: batch must have the following attributes:
            inputs_word, inputs_char, labels
        """
        hidden = self._embed_compute(batch)
        predictions = None
        if compute_predictions:
            predictions = self.output_layer(hidden)
            # probs = F.softmax(tag_logits, -1)
            # _, predictions = torch.max(probs, dim=2)

        # tag_logits = F.log_softmax(tag_logits, dim=-1)
        # loss_val = self.loss_func( tag_logits.view(-1, tag_logits.shape[-1]),
        #                            batch.labels.view(-1))
        loss_val = self.output_layer.loss(hidden, batch.labels)

        return loss_val, predictions

    def compute(self, inputs_word_emb, inputs_char_emb):
        """
        Abstract method that is called to compute the final model
        hidden state. Derived classes implement the method to take
        input embeddings and provide the final hidden state

        Parameters:
            inputs_word_emb: Input word embeddings of shape
                                [batch, sequence-length, word-embedding-size]
            inputs_char_emb[optional]: Input character embeddings of shape
                                [batch x sequence-length, word-length, char-embedding-size]

        Returns:
            Final hidden state in the shape [batch, sequence-length, hidden-size]
        """
        raise NotImplementedError("Must implement compute()")

    @classmethod
    def create(cls, task_name, hparams, vocabs, **kwargs):
        """
        Saves the vocab files
        """
        model = super(Tagger, cls).create(task_name, hparams, vocabs=vocabs, **kwargs)
        model_dir = gen_model_dir(task_name, cls)
        torch.save(vocabs, os.path.join(model_dir, VOCABS_FILE))

        return model

    @classmethod
    def load(cls, task_name, checkpoint, **kwargs):
        model_dir = gen_model_dir(task_name, cls)
        vocabs_path = os.path.join(model_dir, VOCABS_FILE)
        if not os.path.exists(vocabs_path):
            raise OSError('Vocabs file not found')
        vocabs = torch.load(vocabs_path)
        return super(Tagger, cls).load(task_name, checkpoint, vocabs=vocabs, **kwargs)


def hparams_tagging_base():
    return HParams(
        batch_size=100,
        embedding_size_word=300,
        embedding_size_char=0, # No char embeddings
        embedding_size_char_per_word=100,
        embedding_size_tags=100,
        hidden_size=128,
        learning_rate=0.2,
        learning_rate_decay=None,
        max_length=256,
        num_hidden_layers=1,
        dropout=0.2,
        optimizer_adam_beta1=0.9,
        optimizer_adam_beta2=0.98,
        use_crf=False
    )

class TransformerTagger(Tagger):
    """
    Sequence tagger using the Transformer network (https://arxiv.org/pdf/1706.03762.pdf)
    Specifically it uses the Encoder module. For character embeddings (per word) it uses
    the same Encoder module above which an additive (Bahdanau) self-attention layer is added
    """
    def __init__(self, hparams=None, **kwargs):
        """
        No additional parameters
        """
        super(TransformerTagger, self).__init__(hparams=hparams, **kwargs)

        embedding_size = hparams.embedding_size_word
        if hparams.embedding_size_char > 0:
            embedding_size += hparams.embedding_size_char_per_word
            self.transformer_char = transformer.Encoder(
                                        hparams.embedding_size_char,
                                        hparams.embedding_size_char_per_word,
                                        1,
                                        4,
                                        hparams.attention_key_channels,
                                        hparams.attention_value_channels,
                                        hparams.filter_size_char,
                                        hparams.max_length,
                                        hparams.input_dropout,
                                        hparams.dropout,
                                        hparams.attention_dropout,
                                        hparams.relu_dropout,
                                        use_mask=False
                                    ) 
            self.char_linear = nn.Linear(hparams.embedding_size_char_per_word, 1)

        self.transformer_enc = transformer.Encoder(
                                    embedding_size,
                                    hparams.hidden_size,
                                    hparams.num_hidden_layers,
                                    hparams.num_heads,
                                    hparams.attention_key_channels,
                                    hparams.attention_value_channels,
                                    hparams.filter_size,
                                    hparams.max_length,
                                    hparams.input_dropout,
                                    hparams.dropout,
                                    hparams.attention_dropout,
                                    hparams.relu_dropout,
                                    use_mask=False
                                )

    def compute(self, inputs_word_emb, inputs_char_emb):

        if inputs_char_emb is not None:
            seq_len = inputs_word_emb.shape[1]

            # Process character embeddings to get per word embeddings
            inputs_char_emb = self.transformer_char(inputs_char_emb)

            # Apply additive self-attention to combine outputs
            mask = self.char_linear(inputs_char_emb)
            mask = F.softmax(mask, dim=-1)
            inputs_emb_char = (torch.matmul(mask.permute(0, 2, 1), inputs_char_emb).contiguous()
                            .view(-1, seq_len, self.hparams.embedding_size_char_per_word))

            # Combine embeddings
            inputs_word_emb = torch.cat([inputs_word_emb, inputs_emb_char], -1)

        # Apply Transformer Encoder
        enc_out = self.transformer_enc(inputs_word_emb)

        return enc_out

def hparams_transformer_ner():
    hparams = hparams_tagging_base()
    return hparams.update(
        embedding_size_char=16,
        embedding_size_char_per_word=100,
        num_heads=4,
        attention_key_channels=0, # Take hidden size
        attention_value_channels=0, # Take hidden size
        filter_size = 128,
        filter_size_char = 64,
        input_dropout=0.2,
        attention_dropout=0.2,
        relu_dropout=0.2,
        learning_rate_decay='noam_step',
        learning_rate_warmup_steps=500,
        use_crf=True
    )

class BiLSTMTagger(Tagger):
    """
    Sequence tagger using bidirectional LSTM. For character embeddings per word
    uses (unidirectional) LSTM
    """
    def __init__(self, hparams=None, **kwargs):
        """
        No additional parameters
        """
        super(BiLSTMTagger, self).__init__(hparams=hparams, **kwargs)

        embedding_size = hparams.embedding_size_word
        if hparams.embedding_size_char > 0:
            embedding_size += hparams.embedding_size_char_per_word
            self.lstm_char = nn.LSTM(hparams.embedding_size_char, 
                                     hparams.embedding_size_char_per_word,
                                     num_layers=1, 
                                     batch_first=True, 
                                     bidirectional=False)


        self.lstm = nn.LSTM(embedding_size, 
                            hparams.hidden_size, 
                            num_layers=hparams.num_hidden_layers,
                            batch_first=True,
                            bidirectional=True
                           )

        self.dropout = nn.Dropout(hparams.dropout)
        self.extra_layer = nn.Linear(2*hparams.hidden_size, hparams.hidden_size)

    def compute(self, inputs_word_emb, inputs_char_emb):
         
        if inputs_char_emb is not None:
            seq_len = inputs_word_emb.shape[1]

            # Compute per word character embeddings using unidirectional LSTM
            _, (h_n, _) = self.lstm_char(inputs_char_emb)
        
            inputs_char_emb = h_n.view(-1, seq_len, 
                                  self.hparams.embedding_size_char_per_word)

            # Combine embeddings
            inputs_word_emb = torch.cat([inputs_word_emb, inputs_char_emb], -1)
        
        hidden, _ = self.lstm(self.dropout(inputs_word_emb))
        hidden_extra = F.tanh(self.extra_layer(hidden))

        return hidden_extra

def hparams_lstm_ner():
    hparams = hparams_tagging_base()

    return hparams.update(
        embedding_size_char=25,
        embedding_size_char_per_word=25,
        hidden_size=100,
        learning_rate=0.05,
        learning_rate_decay='noam_step',
        learning_rate_warmup_steps=100,
        num_hidden_layers=1,
        dropout=0.5,
        use_crf=True
    )