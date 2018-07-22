from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchnlp.common.hparams import HParams
from torchnlp.common.model import Model, gen_model_dir
from torchnlp.modules import outputs

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

    def forward(self, batch):
        """
        NOTE: batch must have the following attributes:
            inputs_word, inputs_char, labels
        """
        with torch.no_grad():
            hidden = self._embed_compute(batch)
            output = self.output_layer(hidden)

        return output

        # TODO: Add beam search somewhere :)
        

    def loss(self, batch, compute_predictions=False):
        """
        NOTE: batch must have the following attributes:
            inputs_word, inputs_char, labels
        """
        hidden = self._embed_compute(batch)
        predictions = None
        if compute_predictions:
            predictions = self.output_layer(hidden)

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