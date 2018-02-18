# TorchNLP
TorchNLP is a deep learning library for NLP tasks. Built on PyTorch and TorchText, it is an attempt to provide reusable components that work across tasks. Currently it can be used for the Named Entity Recognition (NER) task with a Bidirectional LSTM CRF model and a Transformer network model. It can support any dataset which uses the Conll 2003 format. More tasks will be added shortly

## High Level Workflow
1. Define the NLP task
2. Extend the `Model` class and implement the `forward()` and `loss()` methods to return predictions and loss respectively
3. Use the `HParams` class to easily define the hyperparameters for the model
4. Define a data function to return dataset iterators, vocabularies etc using [TorchText](https://github.com/pytorch/text) API. Check conll.py for an example
5. Set up the `Evaluator` and `Trainer` classes to use the model, dataset iterators and metrics. Check ner.py for details
6. Run the trainer for desired number of epochs along with an early stopping criteria
7. Use the evaluator to evaluate the trained model on a specific dataset split
8. Run inference on the trained model using available input processors

## Boilerplate Components
* `Model`: Handles loading and saving of models as well as the associated hyperparameters
* `HParams`: Generic class to define hyperparameters. Can be persisted
* `Trainer`: Train a given model on a dataset. Supports features like predefined learning rate decay schedules and early stopping
* `Evaluator`: Evaluates the model on a dataset and multiple predefined or custom metrics. 
* `get_input_processor_words`: Use during inference to quickly convert input strings into a format that can be processed by a model

## Available Models
* `transformer.Encoder`, `transformer.Decoder`: Transfomer network implementation from [Attention is all you need](https://arxiv.org/abs/1706.03762)
* `CRF`: Conditional Random Field layer which can be used as the final output
* `TransformerTagger`: Sequence tagging model implemented using the Transformer network and CRF
* `BiLSTMTagger`: Sequence tagging model implemented using bidirectional LSTMs and CRF

## Installation
TBA

## Usage
TorchNLP is designed to be used inside the python interpreter to make it easier to experiment without typing cumbersome command line arguments. 

**NER Task**

TBA
