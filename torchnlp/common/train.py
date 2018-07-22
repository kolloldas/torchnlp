from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from .model import gen_model_dir

import os
from functools import partial
from collections import deque, defaultdict

import logging
logger = logging.getLogger(__name__)

OPTIMIZER_FILE = "optimizer.pt"

# Decay functions to be used with lr_scheduler
def lr_decay_noam(hparams):
    return lambda t: (
        10.0 * hparams.hidden_size**-0.5 * min(
        (t + 1) * hparams.learning_rate_warmup_steps**-1.5, (t + 1)**-0.5))

def lr_decay_exp(hparams):
    return lambda t: hparams.learning_rate_falloff ** t


# Map names to lr decay functions
lr_decay_map = {
    'noam': lr_decay_noam,
    'exp': lr_decay_exp
}

        
def compute_num_params(model):
    """
    Computes number of trainable and non-trainable parameters
    """
    sizes = [(np.array(p.data.size()).prod(), int(p.requires_grad)) for p in model.parameters()]
    return sum(map(lambda t: t[0]*t[1],sizes)), sum(map(lambda t: t[0]*(1 - t[1]),sizes))

class Trainer(object):
    """
    Class to handle training in a task-agnostic way
    """
    def __init__(self, task_name, model, hparams, train_iter, evaluator):
        """
        Parameters:
            task_name: Name of the task
            model: Model instance (derived from model.Model)
            hparams: Instance of HParams
            train_iter: An instance of torchtext.data.Iterator
            evaluator: Instance of evalutation.Evaluator that will
                        run metrics on the validation dataset
        """

        self.task_name = task_name
        self.model = model
        self.hparams = hparams
        self.evaluator = evaluator

        self.train_iter = train_iter
        # Disable repetitions
        self.train_iter.repeat = False

        model_params = filter(lambda p: p.requires_grad, model.parameters())
        
        # TODO: Add support for other optimizers
        self.optimizer = optim.Adam(
                            model_params,
                            betas=(hparams.optimizer_adam_beta1, hparams.optimizer_adam_beta2), 
                            lr=hparams.learning_rate)

        self.opt_path = os.path.join(gen_model_dir(task_name, model.__class__), 
                                    OPTIMIZER_FILE)

        # If model is loaded from a checkpoint restore optimizer also
        if int(model.iterations) > 0:
            self.optimizer.load_state_dict(torch.load(self.opt_path))

        self.lr_scheduler_step = self.lr_scheduler_epoch = None

        # Set up learing rate decay scheme
        if hparams.learning_rate_decay is not None:
            if '_' not in hparams.learning_rate_decay:
                raise ValueError("Malformed learning_rate_decay")
            lrd_scheme, lrd_range = hparams.learning_rate_decay.split('_')

            if lrd_scheme not in lr_decay_map:
                raise ValueError("Unknown lr decay scheme {}".format(lrd_scheme))
            
            lrd_func = lr_decay_map[lrd_scheme]            
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                                            self.optimizer, 
                                            lrd_func(hparams),
                                            last_epoch=int(self.model.iterations) or -1
                                        )
            # For each scheme, decay can happen every step or every epoch
            if lrd_range == 'epoch':
                self.lr_scheduler_epoch = lr_scheduler
            elif lrd_range == 'step':
                self.lr_scheduler_step = lr_scheduler
            else:
                raise ValueError("Unknown lr decay range {}".format(lrd_range))

        # Display number of parameters
        logger.info('Parameters: {}(trainable), {}(non-trainable)'.format(*compute_num_params(self.model)))
    
    def _get_early_stopping_criteria(self, early_stopping):
        es = early_stopping.split('_')
        if len(es) != 3:
            raise ValueError('Malformed early stopping criteria')
        best_type, window, metric = es
        logger.info('Early stopping for {} value of validation {} after {} epochs'
                    .format(best_type, metric, window))
        
        if best_type == 'lowest':
            best_fn = partial(min, key=lambda item: item[0])
        elif best_type == 'highest':
            best_fn = partial(max, key=lambda item: item[0])
        else:
            raise ValueError('Unknown best type {}'.format(best_type))

        return best_fn, int(window), metric

    def train(self, num_epochs, early_stopping=None, save=True):
        """
        Run the training loop for given number of epochs. The model
        is evaluated at the end of every epoch and saved as well
        Parameters:
            num_epochs: Total number of epochs to run
            early_stopping: A string indicating how to perform early stopping
                Should be of the form lowest/highest_n_metric where:
                    lowest/highest: Track lowest or highest values
                    n: The window size within which to track best
                    metric: Name of the metric to track. Should be available
                        in the dict returned by evaluator
            save: Save model every epoch if true
        Returns:
            Tuple of best checkpoint number and metrics array (for plotting etc)
        """

        all_metrics = defaultdict(list)
        best_iteration = 0

        if early_stopping:
            if not save:
                raise ValueError('save should be True for early stopping')
            if self.evaluator is None:
                raise ValueError('early stopping requires an eval function')

            best_fn, best_window, best_metric_name = self._get_early_stopping_criteria(early_stopping)
            tracking = deque([], best_window + 1)


        for epoch in range(num_epochs):
            self.train_iter.init_epoch()
            epoch_loss = 0
            count = 0

            logger.info('Epoch %d (%d)'%(epoch + 1, int(self.model.iterations)))

            prog_iter = tqdm(self.train_iter, leave=False)
            for batch in prog_iter:
                # Train mode
                self.model.train()

                self.optimizer.zero_grad()
                loss, _ = self.model.loss(batch)
                loss.backward()
                self.optimizer.step()

                if self.lr_scheduler_step:
                    self.lr_scheduler_step.step()

                epoch_loss += loss.item()
                count += 1
                self.model.iterations += 1

                # Display loss
                prog_iter.set_description('Training')
                prog_iter.set_postfix(loss=(epoch_loss/count))

            
            if self.lr_scheduler_epoch:
                self.lr_scheduler_epoch.step()

            train_loss = epoch_loss/count
            all_metrics['train_loss'].append(train_loss)
            logger.info('Train Loss: {:3.5f}'.format(train_loss))

            best_iteration = int(self.model.iterations)

            # Run evaluation
            if self.evaluator:
                eval_metrics = self.evaluator.evaluate(self.model)
                if not isinstance(eval_metrics, dict):
                    raise ValueError('eval_fn should return a dict of metrics')

                # Display eval metrics
                logger.info('Validation metrics: ')
                logger.info(', '.join(['{}={:3.5f}'.format(k, v) for k,v in eval_metrics.items()]))

                # Append metrics
                for k, v in eval_metrics.items():
                    all_metrics[k].append(v)

                # Handle early stopping
                tracking.append((eval_metrics[best_metric_name], int(self.model.iterations), epoch))
                logger.debug('Epoch {} Tracking: {}'.format(epoch, tracking))
                
                if epoch >= best_window:
                    # Get the best value of metric in the window
                    best_metric, best_iteration, best_epoch = best_fn(tracking)
                    if tracking[0][1] == best_iteration:
                        # The best value has gone outside the desired window
                        # hence stop
                        logger.info('Early stopping at iteration {}, epoch {}, {}={:3.5f}'
                                    .format(best_iteration, best_epoch, best_metric_name, best_metric))
                        # Update the file time of that checkpoint file to latest
                        self.model.set_latest(self.task_name, best_iteration)
                        break
                
            if save:
                self.model.save(self.task_name)
                torch.save(self.optimizer.state_dict(), self.opt_path)

        return best_iteration, all_metrics            

            
                