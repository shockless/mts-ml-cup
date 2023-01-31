import math

import torch


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0 / math.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_normal(m):
    '''
    Takes in a module and initializes all linear layers with weight
    values taken from a normal distribution.
    '''

    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1 / math.sqrt(y))
        m.bias.data.fill_(0)


def weights_init_xavier(m):
    '''
    Xavier uniform
    '''

    classname = m.__class__.__name__

    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
