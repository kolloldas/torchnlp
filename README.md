# TorchNLP
TorchNLP is a deep learning library for NLP tasks. Built on PyTorch and TorchText, it is an attempt to provide reusable components that work across tasks. Currently it can be used for Named Entity Recognition (NER) and Chunking tasks with a Bidirectional LSTM CRF model and a Transformer network model. It can support any dataset which uses the [CoNLL 2003 format](https://www.clips.uantwerpen.be/conll2003/ner/). More tasks will be added shortly

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
TorchNLP requires a minimum of Python 3.5 and PyTorch 0.4.0 to run. Check [Pytorch](http://pytorch.org/) for the installation steps. 
Clone this repository and install other dependencies like TorchText:
```
pip install -r requirements.txt
```
Go to the root of the project and check for integrity with PyTest:
```
pytest
```
Install this project:
```
python setup.py
```

## Usage
TorchNLP is designed to be used inside the python interpreter to make it easier to experiment without typing cumbersome command line arguments. 

**NER Task**

The NER task can be run on any dataset that confirms to the [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/) format. To use the CoNLL 2003 NER dataset place the dataset files in the following directory structure within your workspace root:
```
.data
  |
  |---conll2003
          |
          |---eng.train.txt
          |---eng.testa.txt
          |---eng.testb.txt
```
`eng.testa.txt` is used the validation dataset and `eng.testb.txt` is used as the test dataset.

Start the NER module in the python shell which sets up the imports:
```
python -i -m torchnlp.ner
```
```
Task: Named Entity Recognition

Available models:
-------------------
TransformerTagger

    Sequence tagger using the Transformer network (https://arxiv.org/pdf/1706.03762.pdf)
    Specifically it uses the Encoder module. For character embeddings (per word) it uses
    the same Encoder module above which an additive (Bahdanau) self-attention layer is added

BiLSTMTagger

    Sequence tagger using bidirectional LSTM. For character embeddings per word
    uses (unidirectional) LSTM


Available datasets:
-------------------
    conll2003: Conll 2003 (Parser only. You must place the files)

>>>
```

Train the [Transformer](https://arxiv.org/abs/1706.03762) model on the CoNLL 2003 dataset:
```
>>> train('ner-conll2003', TransformerTagger, conll2003)
```
The first argument is the task name. You need to use the same task name during evaluation and inference. By default the train function will use the F1 metric with a window of 5 epochs to perform early stopping. To change the early stopping criteria set the `PREFS` global variable as follows:
```
>>> PREFS.early_stopping='lowest_3_loss'
```
This will now use validation loss as the stopping criteria with a window of 3 epochs. The model files are saved under *taskname-modelname* directory. In this case it is *ner-conll2003-TransformerTagger*

Evaluate the trained model on the *testb* dataset split:
```
>>> evaluate('ner-conll2003', TransformerTagger, conll2003, 'test')
```
It will display metrics like accuracy, sequence accuracy, F1 etc

Run the trained model interactively for the ner task:
```
>>> interactive('ner-conll2003', TransformerTagger)
...
Ctrl+C to quit
> Tom went to New York
I-PER O O I-LOC I-LOC
```
You can similarly train the bidirectional LSTM CRF model by using the `BiLSTMTagger` class.
Customizing hyperparameters is quite straight forward. Let's look at the hyperparameters for `TransformerTagger`:
```
>>> h2 = hparams_transformer_ner()
>>> h2

Hyperparameters:
 filter_size=128
 optimizer_adam_beta2=0.98
 learning_rate=0.2
 learning_rate_warmup_steps=500
 input_dropout=0.2
 embedding_size_char=16
 dropout=0.2
 hidden_size=128
 optimizer_adam_beta1=0.9
 embedding_size_word=300
 max_length=256
 attention_dropout=0.2
 relu_dropout=0.2
 batch_size=100
 num_hidden_layers=1
 attention_value_channels=0
 attention_key_channels=0
 use_crf=True
 embedding_size_tags=100
 learning_rate_decay=noam_step
 embedding_size_char_per_word=100
 num_heads=4
 filter_size_char=64
 ```
 Now let's disable the CRF layer:
 ```
 >>> h2.update(use_crf=False)

Hyperparameters:
 filter_size=128
 optimizer_adam_beta2=0.98
 learning_rate=0.2
 learning_rate_warmup_steps=500
 input_dropout=0.2
 embedding_size_char=16
 dropout=0.2
 hidden_size=128
 optimizer_adam_beta1=0.9
 embedding_size_word=300
 max_length=256
 attention_dropout=0.2
 relu_dropout=0.2
 batch_size=100
 num_hidden_layers=1
 attention_value_channels=0
 attention_key_channels=0
 use_crf=False
 embedding_size_tags=100
 learning_rate_decay=noam_step
 embedding_size_char_per_word=100
 num_heads=4
 filter_size_char=64
 ```
 Use it to re-train the model:
 ```
 >>> train('ner-conll2003-nocrf', TransformerTagger, conll2003, hparams=h2)
 ```
 Along with the model the hyperparameters are also saved so there is no need to pass the `HParams` object during evaluation. Also note that by default it will not overwrite any existing model directories (will rename instead). To change that behavior set the PREFS variable:
 ```
 >>> PREFS.overwrite_model_dir = True
 ```
 The `PREFS` variable is automatically persisted in `prefs.json`
 
 **Chunking Task**
 
 The [CoNLL 2000](https://www.clips.uantwerpen.be/conll2000/chunking/) dataset is available for the Chunking task. The dataset is automatically downloaded from the public repository so you don't need to manually download it.

Start the Chunking task:
```
python -i -m torchnlp.chunk
```
Train the [Transformer](https://arxiv.org/abs/1706.03762) model:
```
>>> train('chunk-conll2000', TransformerTagger, conll2000)
```
There is no validation partition provided in the repository hence 10% of the training set is used for validation.

Evaluate the model on the test set:
```
>>> evaluate('chunk-conll2000', TransformerTagger, conll2000, 'test')
```

 ## Standalone Use
 The `transformer.Encoder`, `transformer.Decoder` and `CRF` modules can be independently imported as they only depend on PyTorch:
 ```
 from torchnlp.modules.transformer import Encoder
 from torchnlp.modules.transformer import Decoder
 from torchnlp.modules.crf import CRF
 ```
Please refer to the comments within the source code for more details on the usage
