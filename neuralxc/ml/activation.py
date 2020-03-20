from abc import ABC, abstractmethod

import numpy as np

from ..base import ABCRegistry


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

    def __init__(self):
        pass

    def f(self, x):
        return 1 / (1 + np.exp(-x))

    def df(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2


class Tanh(BaseActivation):

    _registry_name = 'tanh'

    def __init__(self):
        pass

    def f(self, x):
        return np.tanh(x)

    def df(self, x):
        return (1 - np.tanh(x)**2)


def get_activation(activation):
    act_dict = {'sigmoid': Sigmoid, 'tanh': Tanh}
    if not activation in act_dict:
        raise NotImplementedError('This activation function was not implemented')
    return act_dict[activation]()
