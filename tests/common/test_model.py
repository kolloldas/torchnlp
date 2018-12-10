from torchnlp.common.model import gen_model_dir, prepare_model_dir
from torchnlp.common.model import Model, HYPERPARAMS_FILE, CHECKPOINT_FILE

from torchnlp.common.hparams import HParams

import torch
import torch.nn as nn
import os
from time import sleep

class DummyModel(Model):
    def __init__(self, hparams=None, extra=None):
        super(DummyModel, self).__init__(hparams)
        self.extra = extra
        self.param = nn.Parameter(torch.LongTensor([0]), requires_grad=False)

    def loss(self, batch):
        return -1

def test_gen_model_dir(tmpdir):
    tmpdir.chdir()
    model_dir = gen_model_dir('test.Task', Model)
    assert tmpdir.join('test.Task-Model').fnmatch(model_dir)
    assert os.path.exists(model_dir)

def test_prepare_model_dir(tmpdir):
    tmpdir.chdir()
    sub = tmpdir.mkdir('model')

    # Test clearing
    sub.join('dummy.pt').write('x')
    prepare_model_dir(str(sub), True)
    assert len(sub.listdir()) == 0

    # Test rename
    tmpdir.mkdir('model-1')
    sub.join('dummy.pt').write('x')
    prepare_model_dir(str(sub), False)
    assert sub.check()
    assert len(sub.listdir()) == 0
    
    assert tmpdir.join('model-1').check()
    assert len(tmpdir.join('model-1').listdir()) == 0

    assert tmpdir.join('model-2').check()
    assert tmpdir.join('model-2').join('dummy.pt').check()

def test_create_model(tmpdir):
    tmpdir.chdir()

    model = DummyModel.create('test.Task', HParams(test=21), extra=111)
    assert isinstance(model, DummyModel)
    assert hasattr(model, 'hparams')
    assert model.hparams.test == 21
    assert model.extra == 111

def test_load_model(tmpdir):
    tmpdir.chdir()
    sub = tmpdir.mkdir('test.Task-DummyModel')

    torch.save(HParams(test=22), str(sub.join(HYPERPARAMS_FILE)))
    assert sub.join(HYPERPARAMS_FILE).check()

    torch.save(DummyModel(HParams(test=20)).state_dict(), str(sub.join(CHECKPOINT_FILE.format(1))))
    assert sub.join(CHECKPOINT_FILE.format(1)).check()

    sleep(1) # To ensure different file mtimes

    dummy_model = DummyModel(HParams(test=21))
    dummy_model.param += 1
    torch.save(dummy_model.state_dict(), str(sub.join(CHECKPOINT_FILE.format(2))))
    assert sub.join(CHECKPOINT_FILE.format(2)).check()

    model, _ = DummyModel.load('test.Task', checkpoint=-1, extra=111)
    assert isinstance(model, DummyModel)
    assert hasattr(model, 'hparams')
    assert model.hparams.test == 22
    assert model.extra == 111
    assert int(model.param) == 1

def test_save_model(tmpdir):
    tmpdir.chdir()

    dummy_model = DummyModel.create('test.Task', HParams(test=21))
    dummy_model.param += 1
    dummy_model.iterations += 100

    dummy_model.save('test.Task')
    sub = tmpdir.join('test.Task-DummyModel')
    assert sub.check()
    assert sub.join(CHECKPOINT_FILE.format(100)).check()
    assert sub.join(HYPERPARAMS_FILE).check()

    hparams = torch.load(str(sub.join(HYPERPARAMS_FILE)))
    assert isinstance(hparams, HParams)
    assert hparams.test == 21
