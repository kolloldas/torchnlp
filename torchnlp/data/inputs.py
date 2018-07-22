from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torchtext import data

import numpy as np
import logging
logger = logging.getLogger(__name__)

def get_input_processor_words(vocab_word, vocab_char=None, convert_digits=True):
    """
    Returns a function that converts text into a processed batch. Required duing
    inference.
    Parameters:
        vocab_word: Instance of torchtext.Vocab for input word vocabulary
        vocab_char[optional]: Instance of torchtext.Vocab for input per-word 
                              character vocabulary
        convert_digits: If True will convert numbers to single 0's
    """
    inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, lower=True,
                                preprocessing=data.Pipeline(
                                    lambda w: '0' if convert_digits and w.isdigit() else w ))
    # Set the vocab object manually without building from training dataset
    inputs_word.vocab = vocab_word

    if vocab_char is not None:
        inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>", 
                                        batch_first=True)

        inputs_char = data.NestedField(inputs_char_nesting, 
                                        init_token="<bos>", eos_token="<eos>")
        # Set the vocab object manually without building from training dataset
        inputs_char.vocab = inputs_char_nesting.vocab = vocab_char
        
        fields = [(('inputs_word', 'inputs_char'), (inputs_word, inputs_char))]
    else:
        fields = [('inputs_word', inputs_word)]


    def input_processor_fn(inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]

        examples = []
        for line in inputs:
            examples.append(data.Example.fromlist([line], fields))
        
        dataset = data.Dataset(examples, fields)
        # Entire input in one batch
        return data.Batch(data=dataset, 
                          dataset=dataset,
                          device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    return input_processor_fn