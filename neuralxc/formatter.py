import numpy as np
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class Formatter(TransformerMixin, BaseEstimator):

    def __init__(self, basis = None):
        """ Formatter allows to convert between the np.ndarray and the dictionary
        format of basis set representations
        """
        self._basis = basis
        self._rule = None

    def fit(self, C):
        """ Providing C in dictionary format, build set of rules for transformation
        """

        self._rule = {}
        for idx, key, data in expand(C):
            self._rule[key] = [s for s in data[0]]

    def transform(self, C):
        """ Transforms from a dictionary format ({n,l,m} : value)
         that is used internally by neuralxc to an ordered np.ndarray format
        """
        self.fit(C)
        transformed = [{}]*len(C)

        for idx, key, data in expand(C):
            data = data[0]
            if not key in transformed[idx]: transformed[idx][key] = []
            for d in data:
                transformed[idx][key].append(np.array([d[s] for s in d]))

            transformed[idx][key] = np.array(transformed[idx][key])
        if not isinstance(C, list):
            return transformed[0]
        else:
            return transformed

    def inverse_transform(self, C):
        """ Transforms from an ordered np.ndarray format to a dictionary
        format ({n,l,m} : value) that is used internally by neuralxc
        """
        transformed = [{}]*len(C)

        for idx, key, data in expand(C):
            data = data[0]
            if not key in transformed[idx]: transformed[idx][key] = []


            if not isinstance(self._rule, dict):
                basis = self._basis[key]
                rule = ['{},{},{}'.format(n,l,m) for n in range(0,basis['n'])  for l in range(0,basis['l']) for m in range(-l,l+1)]
            else:
                rule = self._rule[key]
            for d in data:
                transformed[idx][key].append(dict(zip(rule, d.tolist())))

        if not isinstance(C, list):
            return transformed[0]
        else:
            return transformed

def expand(*args):
    """ Takes the common format in which datasets such as D and C are provided
     (usually [{'species': np.ndarray}]) and loops over it
     """
    args = list(args)
    for i, arg in enumerate(args):
        if not isinstance(arg, list):
            args[i] = [arg]

    for idx, datasets in enumerate(zip(*args)):
        for key in datasets[0]:
            yield (idx, key , [data[key] for data in datasets])
