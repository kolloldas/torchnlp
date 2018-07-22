from torchnlp.common.evaluation import convert_iob_to_segments
from torchnlp.common.evaluation import Evaluator, Metrics, BasicMetrics, IOBMetrics
from torchnlp.common.model import Model
from torchnlp.common.hparams import HParams

import torch


class DummyModel(Model):
    def loss(self, batch, compute_predictions=False):
        return torch.FloatTensor([1]), 10

class DummyMetrics(Metrics):
    def __init__(self, mul):
        self.mul = mul

    def reset(self):
        self.total = 0
        self.count = 0
        self.batches = 0

    def evaluate(self, batch, loss, predictions):
        self.total += predictions*self.mul
        self.count += 1
        self.batches += batch

    def results(self, total_loss):
        return {
            'test_{}'.format(self.mul): self.total/self.count,
            'batches': self.batches
        }

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

def test_convert_iob_to_segments():
    tags = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG',
           'B-LOC', 'I-LOC', 'O', 'B-WE-IRD', 'I-WE-IRD',
           '<SOS>', '<EOS>', '<UNK>', '<PAD>']
    tag_mapping = {t:i for i,t in enumerate(tags)}

    pairs = [(['<SOS>', '<EOS>'], 
             set([])),

             (['<SOS>', 'B-PER', '<EOS>'], 
             set([('PER', 1, 1)])),

             (['<SOS>', 'B-PER', 'O', 'B-PER', '<EOS>'], 
             set([('PER', 1, 1), ('PER', 3, 3)])),

             (['<SOS>', 'B-PER', 'B-PER', '<EOS>'], 
             set([('PER', 1, 1), ('PER', 2, 2)])),

             (['<SOS>', 'B-PER', 'I-PER', 'I-PER', '<EOS>', '<PAD>'], 
             set([('PER', 1, 3)])),

             (['<SOS>', 'O', 'O', 'B-LOC', 'I-ORG', '<EOS>'], 
             set([('LOC', 3, 3), ('ORG', 4, 4)])),

             (['<SOS>', 'B-PER', 'I-PER', '<UNK>', 'B-LOC', 'I-LOC', '<EOS>'], 
             set([('PER', 1, 2), ('LOC', 4, 5)])),

             (['<SOS>', 'I-PER', 'I-ORG', 'I-LOC', '<EOS>'], 
             set([('PER', 1, 1), ('ORG', 2, 2), ('LOC', 3, 3)])),
             
             (['<SOS>', 'O', 'B-WE-IRD', 'I-WE-IRD', '<EOS>'], 
             set([('WE-IRD', 2, 3)]))]

    for sequence, true_segment in pairs:
        seq_ids = map(lambda item: tag_mapping[item], sequence)
        gen_segment = convert_iob_to_segments(seq_ids, tags)
        assert true_segment == gen_segment

    
def test_evaluator():
    model = DummyModel(hparams=HParams(test=1))
    data_iter = DummyIter(2)
    metric_1, metric_2 = DummyMetrics(1), DummyMetrics(2)

    evaluator = Evaluator(data_iter, metric_1, metric_2)
    result = evaluator.evaluate(model)

    assert isinstance(result, dict)
    assert result['loss'] == 1
    assert result['test_1'] == 10
    assert result['test_2'] == 20
    assert result['batches'] == 3

    result = evaluator.evaluate(model)
    assert result['loss'] == 1
    assert result['test_1'] == 10
    assert result['test_2'] == 20
    assert result['batches'] == 3
    
def test_basic_metrics():
    class Vocab(object):
        def __init__(self):
            self.stoi = {'<pad>': 2, '<unk>': 3}

    class Batch(object):
        def __init__(self):
            self.labels = torch.LongTensor([[0, 1, 2], [0, 0, 3], 
                                                     [0, 0, 0], [1, 1, 1]])

    predictions_1 = torch.LongTensor([[0, 1, 2], [0, 0, 3], 
                                               [0, 0, 0], [1, 1, 1]])

    loss = torch.FloatTensor([0])

    basic = BasicMetrics(Vocab())
    basic.reset()
    basic.evaluate(Batch(), loss, predictions_1)
    result = basic.results(0)

    assert isinstance(result['acc'], float)
    assert result['acc'] == 1
    assert result['acc-seq'] == 1

    predictions_2 = torch.LongTensor([[0, 1, 0], [0, 0, 0], 
                                               [0, 0, 0], [1, 1, 1]])
    basic.reset()
    basic.evaluate(Batch(), loss, predictions_2)
    result = basic.results(0)

    assert isinstance(result['acc'], float)
    assert result['acc'] == 1
    assert result['acc-seq'] == 0.5

    predictions_3 = torch.LongTensor([[0, 0, 0], [0, 0, 0], 
                                    [1, 0, 0], [1, 1, 1]])
    basic.reset()
    basic.evaluate(Batch(), loss, predictions_3)
    result = basic.results(0)

    assert isinstance(result['acc'], float)
    assert result['acc'] == 0.8
    assert result['acc-seq'] == 0.25

def test_iob_metrics():
    class Vocab(object):
        def __init__(self):
            self.itos = ['B-PER', 'I-PER', 'B-ORG', 'I-ORG',
                        'B-LOC', 'I-LOC', 'O',
                        '<SOS>', '<EOS>', '<UNK>', '<PAD>']

    class Batch(object):
        def __init__(self):
            self.labels = torch.LongTensor([[0, 1, 6], [0, 0, 1], 
                                            [0, 2, 3], [6, 6, 6]])

    # All Correct
    predictions_1 = torch.LongTensor([[0, 1, 6], [0, 0, 1], 
                                    [0, 2, 3], [6, 6, 6]])

    loss = torch.FloatTensor([0])

    iob = IOBMetrics(Vocab())
    iob.reset()
    iob.evaluate(Batch(), loss, predictions_1)
    result = iob.results(0)

    assert isinstance(result['precision'], float)
    assert isinstance(result['recall'], float)
    assert isinstance(result['F1'], float)
    assert result['precision'] == 1
    assert result['recall'] == 1
    assert result['F1'] == 1

    # Ignore anything not B/I
    predictions_2 = torch.LongTensor([[0, 1, 6], [0, 0, 1], 
                                    [0, 2, 3], [9, 9, 9]])

    loss = torch.FloatTensor([0])

    iob = IOBMetrics(Vocab())
    iob.reset()
    iob.evaluate(Batch(), loss, predictions_2)
    result = iob.results(0)

    assert result['precision'] == 1
    assert result['recall'] == 1
    assert result['F1'] == 1

    # Lower precision/F1
    predictions_3 = torch.LongTensor([[0, 0, 10], [0, 0, 1], 
                                        [0, 2, 3], [0, 0, 9]])

    loss = torch.FloatTensor([0])

    iob = IOBMetrics(Vocab())
    iob.reset()
    iob.evaluate(Batch(), loss, predictions_3)
    result = iob.results(0)

    assert result['precision'] == 0.5
    assert result['recall'] == 0.8
    assert result['F1'] - 0.6153 < 1e-4
