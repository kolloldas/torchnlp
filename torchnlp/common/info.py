from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class Info(object):
    """
    Displays information about models and datasets.
    Currently a stub
    """
    def __init__(self, docstring):
        print('\nTask:', docstring.strip())

    def models(self, *args):
        print('\nAvailable models:')
        print('-------------------')
        print('\n'.join('{}\n{}'.format(m.__name__, m.__doc__) for m in args))
        
        return self

    def datasets(self, *args):
        print('\nAvailable datasets:')
        print('-------------------')
        print('\n'.join(d.__doc__.split('\n')[1] for d in args if d.__doc__))
        print()
        return self
