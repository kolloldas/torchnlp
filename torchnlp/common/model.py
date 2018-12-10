from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import os
import glob

import logging
logger = logging.getLogger(__name__)

HYPERPARAMS_FILE = 'hyperparams.pt'
CHECKPOINT_FILE = 'checkpoint-{}.pt'
CHECKPOINT_GLOB = 'checkpoint-*.pt'

def gen_model_dir(task_name, model_cls):
    """
    Generate the model directory from the task name and model class.
    Creat if not exists. 
    Parameters:
        task_name: Name of the task. Gets prefixed to model directory
        model_cls: The models class (derived from Model)
    """
    model_dir = os.path.join(os.getcwd(), '%s-%s'%(task_name, model_cls.__name__))
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    return model_dir

def prepare_model_dir(model_dir, clear=False):
    """
    Prepares the model directory. If clear is set to True, deletes all files, else
    renames existing directory and creates a fresh one
    Parameters:
        model_dir: Absolute path of the model directory to prepare
    """
    p = list(os.walk(model_dir))

    # Check if directory is not empty (ignore subdirectories)
    if clear:
        for file in os.listdir(model_dir):
            # Delete all files
            path = os.path.join(model_dir, file)
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except:
                    logger.warning('WARNING: Failed to delete {}'.format(path))
    elif len(p[0][2]) > 0:
        # Rename to available directory
        # Max index is 10
        for i in range(1, 10):
            try:
                rename_dir = '{}-{}'.format(model_dir, i)
                if not os.path.exists(rename_dir):
                    os.rename(model_dir, rename_dir)
                    os.mkdir(model_dir)
                    break
            except:
                pass
        

def xavier_uniform_init(m):
    """
    Xavier initializer to be used with model.apply
    """
    if type(m) == nn.Linear: # TODO: Add Embeddings?
        nn.init.xavier_uniform_(m.weight.data)

class Model(nn.Module):
    """
    Abstract base class that defines a loss() function
    """
    def __init__(self, hparams=None):
        super(Model, self).__init__()
        
        if hparams is None:
            raise ValueError('Must provide hparams')
        
        self.hparams = hparams

        # Track total iterations
        self.iterations = nn.Parameter(torch.LongTensor([0]), requires_grad=False)

    def loss(self, batch, compute_predictions=False):
        """
        Called by train.Trainer to compute negative log likelihood
        Parameters:
            batch: A minibatch, instance of torchtext.data.Batch
            compute_predictions: If true compute and provide predictions, else None
        Returns:
            Tuple of loss and predictions
        """
        raise NotImplementedError("Must implement loss()")

    @classmethod
    def create(cls, task_name, hparams, overwrite=False, **kwargs):
        """
        Create a new instance of this class. Prepares the model directory
        and saves hyperparams. Derived classes should override this function
        to save other dependencies (e.g. vocabs)
        """
        logger.info(hparams)
        model_dir = gen_model_dir(task_name, cls)
        model = cls(hparams, **kwargs)
        # Xavier initialization
        model.apply(xavier_uniform_init)

        if torch.cuda.is_available():
            model = model.cuda()

        prepare_model_dir(model_dir, overwrite)

        #Save hyperparams
        torch.save(hparams, os.path.join(model_dir, HYPERPARAMS_FILE))

        return model

    @classmethod
    def load(cls, task_name, checkpoint, **kwargs):
        """
        Loads a model from a checkpoint. Also loads hyperparams
        Parameters:
            task_name: Name of the task. Needed to determine model dir
            checkpoint: Number indicating the checkpoint. -1 to load latest
            **kwargs: Additional key-value args passed to constructor
        """
        model_dir = gen_model_dir(task_name, cls)
        hparams_path = os.path.join(model_dir, HYPERPARAMS_FILE)

        if not os.path.exists(hparams_path):
            raise OSError('HParams file not found')

        hparams = torch.load(hparams_path)
        logger.info('Hyperparameters: {}'.format(str(hparams)))

        model = cls(hparams, **kwargs)
        if torch.cuda.is_available():
            model = model.cuda()

        if checkpoint == -1:
            # Find latest checkpoint file
            files = glob.glob(os.path.join(model_dir, CHECKPOINT_GLOB))
            if not files:
                raise OSError('Checkpoint files not found')
            files.sort(key=os.path.getmtime, reverse=True)
            checkpoint_path = files[0]
        else:
            checkpoint_path = os.path.join(model_dir, CHECKPOINT_FILE.format(checkpoint))
            if not os.path.exists(checkpoint_path):
                raise OSError('File not found: {}'.format(checkpoint_path))

        logger.info('Loading from {}'.format(checkpoint_path))
        # Load the model
        model.load_state_dict(torch.load(checkpoint_path))

        return model, hparams

    def save(self, task_name):
        """
        Save the model. Directory is determined by the task name and model class name
        """
        model_dir = gen_model_dir(task_name, self.__class__)
        checkpoint_path = os.path.join(model_dir, CHECKPOINT_FILE.format(int(self.iterations)))
        torch.save(self.state_dict(), checkpoint_path)
        logger.info('-------------- Saved checkpoint {} --------------'.format(int(self.iterations)))

    def set_latest(self, task_name, iteration):
        """
        Set the modified time of the checkpoint to latest. Used to set the best
        checkpoint
        """
        model_dir = gen_model_dir(task_name, self.__class__)
        checkpoint_path = os.path.join(model_dir, CHECKPOINT_FILE.format(int(iteration)))
        os.utime(checkpoint_path, None)
