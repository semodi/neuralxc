import numpy as np
from ..base import ABCRegistry
from abc import ABC, abstractmethod
try:
    import torch
except ModuleNotFoundError:
    pass

def get_lib(lib): #Necessary otherwise pickling not possible
    if lib == 'np':
        return np
    elif lib == 'torch':
        return torch
    else:
        raise Exception('Library {} {} not available'.format(lib,type(lib)))

class BaseActivation(ABC):

    _registry_name = 'base'

    @abstractmethod
    def f(self, x):
        pass

    @abstractmethod
    def df(self, x):
        pass


class Sigmoid(BaseActivation):

    _registry_name = 'sigmoid'

    def __init__(self, lib = 'np'):
        self.lib = lib

    def f(self, x):
        if not hasattr(self, 'lib'):
            self.lib = 'np'

        lib = get_lib(self.lib)
        return 1 / (1 + lib.exp(-x))

    def df(self, x):
        if not hasattr(self, 'lib'):
            self.lib = 'np'
        lib = get_lib(self.lib)
        return lib.exp(-x) / (1 + lib.exp(-x))**2


class Tanh(BaseActivation):

    _registry_name = 'tanh'

    def __init__(self, lib = 'np'):
        self.lib = lib

    def f(self, x):
        if not hasattr(self, 'lib'):
            self.lib = 'np'
        lib = get_lib(self.lib)
        return lib.tanh(x)

    def df(self, x):
        if not hasattr(self, 'lib'):
            self.lib = 'np'
        lib = get_lib(self.lib)
        return (1 - lib.tanh(x)**2)


def get_activation(activation):
    act_dict = {'sigmoid': Sigmoid, 'tanh': Tanh}
    if not activation in act_dict:
        raise NotImplementedError('This activation function was not implemented')
    return act_dict[activation]()
