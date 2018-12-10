from torchnlp.tasks.sequence_tagging import Tagger, hparams_tagging_base, VOCABS_FILE

import torch
import torch.nn as nn

import torchtext
from torchtext import data
from torchtext import datasets

import pytest

def udpos_dataset(batch_size):
    # Setup fields with batch dimension first
    inputs = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)
    tags = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)
    
    # Download and the load default data.
    train, val, test = datasets.UDPOS.splits(
    fields=(('inputs_word', inputs), ('labels', tags), (None, None)))
    
    # Build vocab
    inputs.build_vocab(train.inputs)
    tags.build_vocab(train.tags)
    
    # Get iterators
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
                            (train, val, test), batch_size=batch_size, 
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    train_iter.repeat = False
    return train_iter, val_iter, test_iter, inputs, tags

class DummyTagger(Tagger):
    def __init__(self, hparams, **kwargs):
        super(DummyTagger, self).__init__(hparams=hparams, **kwargs)
        self.linear = nn.Linear(hparams.embedding_size_word, 
                                hparams.hidden_size)

    def compute(self, inputs_word_emb, inputs_char_emb):
        return self.linear(inputs_word_emb)

@pytest.mark.slow
def test_tagger(tmpdir):
    tmpdir.chdir()
    hparams = hparams_tagging_base()
    train_iter, val_iter, test_iter, inputs, tags = udpos_dataset(hparams.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tagger = DummyTagger(hparams=hparams, vocabs=(inputs.vocab, None, tags.vocab)).to(device)
    assert tagger.embedding_word.weight.shape == (len(inputs.vocab), hparams.embedding_size_word)
    assert tagger.output_layer.output_projection.weight.shape == (len(tags.vocab), hparams.hidden_size)

    batch = next(iter(val_iter))
    loss, preds = tagger.loss(batch, compute_predictions=True)

    assert loss > 0
    assert preds.data.shape == batch.labels.data.shape

@pytest.mark.slow
def test_tagger_create(tmpdir):
    tmpdir.chdir()
    hparams = hparams_tagging_base()
    train_iter, val_iter, test_iter, inputs, tags = udpos_dataset(hparams.batch_size)

    tagger = DummyTagger.create('test.Task', hparams=hparams, vocabs=(inputs.vocab, None, tags.vocab))
    assert isinstance(tagger, DummyTagger)
    assert tmpdir.join('test.Task-DummyTagger').join(VOCABS_FILE).check()

@pytest.mark.slow
def test_tagger_load(tmpdir):
    tmpdir.chdir()
    hparams = hparams_tagging_base()
    train_iter, val_iter, test_iter, inputs, tags = udpos_dataset(hparams.batch_size)

    tagger = DummyTagger.create('test.Task', hparams=hparams, vocabs=(inputs.vocab, None, tags.vocab))
    tagger.iterations += 10
    tagger.save('test.Task')

    tagger_load, _ = DummyTagger.load('test.Task', checkpoint=-1)
    assert isinstance(tagger_load.vocab_tags, torchtext.vocab.Vocab)