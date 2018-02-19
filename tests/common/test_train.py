from torchnlp.common.train import Trainer
from torchnlp.common.model import Model, CHECKPOINT_FILE
from torchnlp.common.hparams import HParams, hparams_basic

import torch
import torch.nn as nn

class DummyModel(Model):
    def __init__(self, hparams=None):
        super(DummyModel, self).__init__(hparams)
        self.dummy_param = nn.Parameter(torch.FloatTensor([1]))

    def loss(self, batch):
        return self.dummy_param, 0

class DummyIter(object):
    def __init__(self, tot):
        self.tot = tot

    def init_epoch(self):
        pass

    def __len__(self):
        return self.tot

    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count < self.tot:
            self.count += 1
            return self.count
        else:
            raise StopIteration
    next = __next__

def test_get_early_stopping_criteria(tmpdir):
    tmpdir.chdir()
    model = DummyModel(hparams=hparams_basic())
    trainer = Trainer('test.Task', model, model.hparams, DummyIter(1), None)
    best_fn, window, metric = trainer._get_early_stopping_criteria('lowest_3_loss')

    assert window == 3
    assert metric == 'loss'
    assert best_fn([(1, 2), (3, 4)])[1] == 2


def test_early_stopping(tmpdir):
    tmpdir.chdir()
    model = DummyModel(hparams=hparams_basic())

    class Evaluator(object):
        def __init__(self):
            self.losses = [
                100, 50, 20, 15, 14, 13, 12, 10, 10,
                11, 9, 8, 9, 10, 10, 12, 9, 9, 11
            ]
            self.count = -1

        def evaluate(self, model):
            self.count += 1
            return {'loss': self.losses[self.count]}

    trainer = Trainer('test.Task', model, model.hparams, DummyIter(2), Evaluator())
    best_iteration, _ = trainer.train(20, early_stopping='lowest_5_loss')
    
    assert best_iteration == 24
    assert tmpdir.join('test.Task-DummyModel').join(CHECKPOINT_FILE.format(best_iteration)).check()