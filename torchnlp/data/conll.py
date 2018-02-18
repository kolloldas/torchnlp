from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe, CharNGram

import numpy as np
import logging
logger = logging.getLogger(__name__)

# Temporary class till datasets.SequenceTaggingDataset supports specifying a separator
class SequenceTaggingDataset(data.Dataset):
    """Defines a dataset for sequence tagging. Examples in this dataset
    contain paired lists -- paired list of words and tags.
    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]
    See torchtext/test/sequence_tagging.py on how to use this class.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, separator=' ', **kwargs):
        examples = []
        columns = []

        with open(path) as input_file:
            for line in input_file:
                line = line.strip()
                if line == "":
                    if columns:
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(SequenceTaggingDataset, self).__init__(examples, fields,
                                                     **kwargs)

def conll2003_dataset(tag_type, batch_size, root='./conll2003', 
                          train_file='eng.train.txt', 
                          validation_file='eng.testa.txt',
                          test_file='eng.testb.txt',
                          convert_digits=True):
    """
    conll2003: Conll 2003 (Parser only. You must place the files)
    Extract Conll2003 dataset using torchtext. Applies GloVe 6B.200d and Char N-gram
    pretrained vectors. Also sets up per word character Field
    Parameters:
        tag_type: Type of tag to pick as task [pos, chunk, ner]
        batch_size: Batch size to return from iterator
        train_file: Train filename
        validation_file: Validation filename
        test_file: Test filename
        convert_digits: If True will convert numbers to single 0's

    Returns:
        A dict containing:
            task: 'conll2003.' + tag_type
            iters: (train iter, validation iter, test iter)
            vocabs: (Inputs word vocabulary, Inputs character vocabulary, 
                    Tag vocabulary )
    """
    
    # Setup fields with batch dimension first
    inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, lower=True,
                                preprocessing=data.Pipeline(
                                    lambda w: '0' if convert_digits and w.isdigit() else w ))

    inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", 
                                    batch_first=True)

    inputs_char = data.NestedField(inputs_char_nesting, 
                                    init_token="<bos>", eos_token="<eos>")
                        

    labels = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    fields = ([(('inputs_word', 'inputs_char'), (inputs_word, inputs_char))] + 
                [('labels', labels) if label == tag_type else (None, None) 
                for label in ['pos', 'chunk', 'ner']])

    # Load the data
    train, val, test = SequenceTaggingDataset.splits(
                                path=root, 
                                train=train_file, 
                                validation=validation_file, 
                                test=test_file,
                                fields=tuple(fields))

    logger.info('---------- CONLL 2003 %s ---------'%tag_type)
    logger.info('Train size: %d'%(len(train)))
    logger.info('Validation size: %d'%(len(val)))
    logger.info('Test size: %d'%(len(test)))
    
    # Build vocab
    inputs_char.build_vocab(train.inputs_char, val.inputs_char, test.inputs_char)
    inputs_word.build_vocab(train.inputs_word, val.inputs_word, test.inputs_word, max_size=50000,
                        vectors=[GloVe(name='6B', dim='200'), CharNGram()])
    
    labels.build_vocab(train.labels)
    logger.info('Input vocab size:%d'%(len(inputs_word.vocab)))
    logger.info('Tagset size: %d'%(len(labels.vocab)))

    # Get iterators
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
                            (train, val, test), batch_size=batch_size, 
                            device=0 if torch.cuda.is_available() else -1)
    train_iter.repeat = False
    
    return {
        'task': 'conll2003.%s'%tag_type,
        'iters': (train_iter, val_iter, test_iter), 
        'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab) 
        }
    

