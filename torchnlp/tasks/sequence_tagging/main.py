
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

from torchnlp.common.hparams import HParams
from torchnlp.common.evaluation import Evaluator, BasicMetrics, IOBMetrics
from torchnlp.common.train import Trainer
from torchnlp.data.inputs import get_input_processor_words
from torchnlp.common.prefs import PREFS

import sys, os
import logging

logger = logging.getLogger(__name__)


# Globals
SPECIALS = set(['<unk>', '<pad>', '<bos>', '<eos>'])

def train(task_name, model_cls, dataset_fn, hparams=None, hparams_map=None,
            num_epochs=100, checkpoint=None,
            use_iob_metrics=True, early_stopping=None):
    """
    Train a model on a dataset. Saves model and hyperparams so that training
    can be easily resumed
    Parameters:
        task_name: Name of the task like NER1 etc. More like a unique identifier
        model_cls: Model class name. Should inherit from sequence_tagging.Tagger class
        dataset_fn: Dataset function. Must return a dict. See data/conll.py
        hparams: Hyperparameters defined for the particular model. Not required when
                resuming training
        hparams_map: mapping between model classes and hparams. Only used of hparams is
                None
        num_epochs: Set to 100. With early stopping set number of epochs will usually
                be lower. Set PREFS.early_stopping to desired criteria
        checkpoint: Saved checkpoint number (train iteration). Used to resume training
        use_iob_metrics: If True then adds IOB metrics (precision/recall/F1) to Evaluator
        early_stopping: Early stopping criteria. Should be of the form
                        [lowest/highest]_[window_Size]_[metric] where metric is one of
                        loss, acc, acc-seq, precision, recall, F1. Default is picked
                        from PREFS.early_stopping
    """
    early_stopping=early_stopping or PREFS.early_stopping

    # Create or load the model
    if checkpoint is None:
        if hparams is None:
            hparams = hparams_map[model_cls]
        dataset = dataset_fn(hparams.batch_size)
        model = model_cls.create(task_name, hparams, vocabs=dataset['vocabs'], 
                                overwrite=PREFS.overwrite_model_dir)
    else:
        model, hparams_loaded = model_cls.load(task_name, checkpoint)
        if hparams is None:
            hparams = hparams_loaded
        dataset = dataset_fn(hparams.batch_size)
    
    train_iter, validation_iter, _ = dataset['iters']
    _, _, tag_vocab = dataset['vocabs']

    metrics = [BasicMetrics(output_vocab=tag_vocab)]
    if use_iob_metrics:
        metrics += [IOBMetrics(tag_vocab=tag_vocab)]

    # Setup evaluator on the validation dataset
    evaluator = Evaluator(validation_iter, *metrics)

    # Setup trainer
    trainer = Trainer(task_name, model, hparams, train_iter, evaluator)

    # Start training
    best_cp, _ = trainer.train(num_epochs, early_stopping=early_stopping)

    return best_cp


def evaluate(task_name, model_cls, dataset_fn, split, checkpoint=-1, use_iob_metrics=True):
    """
    Evaluate the model on a specific dataset split [train, validation, test].
    Evaluated for loss, accuracy, precision, recall, F1. Model hyperparameters
    are loaded automatically
    Parameters:
        task_name: Name of the task like NER1 etc. You must use the same name
                    that was provided during training
        model_cls: Model class name. Must be same as used during training
        dataset_fn: Dataset function. Must return a dict. See data/conll.py
        split: One of the strings train/validation/test
        checkpoint: Checkpoint number (train iteration) to load from. -1 for
                    best/latest checkpoint
        use_iob_metrics: If True then adds IOB metrics (precision/recall/F1) 
                    to Evaluator
    """
    iter_map = {'train': 0, 'validation': 1, 'test': 2}
    dataset = dataset_fn()
    data_iter = dataset['iters'][iter_map[split]]

    model, hparams = model_cls.load(task_name, checkpoint)

    metrics = [BasicMetrics(output_vocab=model.vocab_tags)]
    if use_iob_metrics:
        metrics += [IOBMetrics(tag_vocab=model.vocab_tags)]

    # Setup evaluator on the given dataset
    evaluator = Evaluator(data_iter, *metrics)
    results = evaluator.evaluate(model)

    print('{} set evaluation: {}-{}'.format(split, task_name, model.__class__.__name__))
    print(', '.join(['{}={:3.5f}'.format(k, v) for k,v in results.items()]))

def _run_model_loaded(model, batch):
    predictions = model(batch)
    results = []
    for j in range(predictions.shape[0]):
        pred_indices = list(predictions[j, :].data)
        tags = map(lambda id: model.vocab_tags.itos[id], pred_indices)
        tags = filter(lambda item: item not in SPECIALS, tags)
        results.append(list(tags))

    return results

def infer(task_name, inputs, model_cls, checkpoint=-1):
    """
    Run inference on an input sentence. 
    Parameters:
        task_name: Name of the task as provided during training
        inputs: Input sentence Either a string or a list of strings
        model_cls: Model class name. Must be same as used during training
        checkpoint: Checkpoint number (train iteration) to load from. -1 for
                    best/latest checkpoint
    """
    task_name = task_name or PREFS.cur_task
    model = model_cls.load(task_name, checkpoint)
    
    vocab_word, vocab_char, _ = model.vocabs
    input_fn = get_input_processor_words(vocab_word, vocab_char)

    batch = input_fn(inputs)
    
    return _run_model_loaded(model, batch)

def interactive(task_name, model_cls, checkpoint=-1):
    """
    Run inference interactively
    Parameters:
        task_name: Name of the task as provided during training
        model_cls: Model class name. Must be same as used during training
        checkpoint: Checkpoint number (train iteration) to load from. -1 for
                    best/latest checkpoint
    """
    model, hparams = model_cls.load(task_name, checkpoint)

    vocab_word, vocab_char, _ = model.vocabs
    input_fn = get_input_processor_words(vocab_word, vocab_char)

    print('Ctrl+C to quit')
    try:
        while True:
            inputs = input('> ')
            batch = input_fn(inputs)
            output = _run_model_loaded(model, batch)
            print(' '.join(output[0]))
    except:
        pass
    

