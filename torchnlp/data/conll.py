from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset, CoNLL2000Chunking
from torchtext.vocab import Vectors, GloVe, CharNGram

import numpy as np
import random
import logging
logger = logging.getLogger(__name__)


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
        root: Dataset root directory
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
                                separator=' ',
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
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    train_iter.repeat = False
    
    return {
        'task': 'conll2003.%s'%tag_type,
        'iters': (train_iter, val_iter, test_iter), 
        'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab) 
        }
    

def conll2000_dataset(batch_size, use_local=False, root='.data/conll2000',
                            train_file='train.txt',
                            test_file='test.txt',
                            validation_frac=0.1,
                            convert_digits=True):
    """
    conll2000: Conll 2000 (Chunking)
    Extract Conll2000 Chunking dataset using torchtext. By default will fetch
    data files from online repository.
    Applies GloVe 6B.200d and Char N-gram pretrained vectors. Also sets 
    up per word character Field
    Parameters:
        batch_size: Batch size to return from iterator
        use_local: If True use local provided files (default False)
        root (optional): Dataset root directory (needed only if use_local is True)
        train_file (optional): Train filename (needed only if use_local is True)
        test_file (optional): Test filename (needed only if use_local is True)
        validation_frac (optional): Fraction of train dataset to use for validation
        convert_digits (optional): If True will convert numbers to single 0's
    NOTE: Since there is only a train and test set we use 10% of the train set as
        validation
    Returns:
        A dict containing:
            task: 'conll2000.' + tag_type
            iters: (train iter, validation iter, None)
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

    fields = [(('inputs_word', 'inputs_char'), (inputs_word, inputs_char)),
                (None, None), ('labels', labels)]

    if use_local:
        # Load the data
        train, test = SequenceTaggingDataset.splits(
                                    path=root, 
                                    train=train_file,
                                    test=test_file,
                                    fields=tuple(fields))

        # HACK: Saving the sort key function as the split() call removes it
        sort_key = train.sort_key
        # To make the split deterministic
        random.seed(0)
        train, val = train.split(1 - validation_frac, random_state=random.getstate())
        # Reset the seed
        random.seed()

        # HACK: Set the sort key
        train.sort_key = sort_key
        val.sort_key = sort_key
    else:
        train, val, test = CoNLL2000Chunking.splits(fields=tuple(fields), 
                                                    validation_frac=validation_frac)

    logger.info('---------- CONLL 2000 Chunking ---------')
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
                            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    train_iter.repeat = False
    
    return {
        'task': 'conll2000.chunk',
        'iters': (train_iter, val_iter, test_iter), 
        'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab) 
        }