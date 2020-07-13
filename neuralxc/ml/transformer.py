""" Implements custom transformer for the special kind of grouped datasets
that neuralxc is working with.

A typical dataset looks like this:
[{'spec1': features,'spec2' : features}, {'spec1': features, 'spec3': features}]
where the outer list runs over independent systems.
"""
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ..formatter import atomic_shape, system_shape
from abc import ABC, abstractmethod
import numpy as np
try:
    import torch
    TorchModule = torch.nn.Module
except ModuleNotFoundError:
    class TorchModule:
         def __init__(self):
             pass

def convert_torch_wrapper(func):

    def wrapped_func(X, *args, **kwargs):
        X = torch.from_numpy(X)
        Y = func(X, *args, **kwargs)
        return Y.detach().numpy()

    return wrapped_func

class GroupedTransformer(ABC):
    """ Abstract base class, grouped transformer extend the functionality
    of sklearn Transformers to neuralxc specific grouped data. Further, they
    implement a get_gradient method.
    """

    #TODO: make _get_gradient abstractmethod and _before_fit an abstractparameter

    def __init__(self, *args, **kwargs):

        self.is_fit = False
        self.is_torch = False
        super().__init__(*args, **kwargs)

    def transform(self, X, y=None, **fit_params):
        if hasattr(self, 'is_torch'):
            if self.is_torch and fit_params.get('wrap_torch', True) and hasattr(self, '_spec_dict'):
                for spec in self._spec_dict:
                    trafo = self._spec_dict[spec]
                    trafo.torch_transform = convert_torch_wrapper(trafo.torch_transform)

        was_tuple = False
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
            was_tuple = True

        made_list = False
        if not isinstance(X, list):
            X = [X]
            made_list = True

        results = []
        for x in X:
            if isinstance(x, dict):
                results_dict = {}
                for spec in x:
                    results_dict[spec] = self._spec_dict[spec].transform(x[spec])
                results.append(results_dict)
            else:
                if hasattr(self,'is_torch') and self.is_torch:
                    results.append(system_shape(self.torch_transform(atomic_shape(x)), x.shape[-2]))
                else:
                    mask = ~np.all(atomic_shape(x) == 0, axis=-1)
                    res = atomic_shape(x)
                    res[mask] = super().transform(atomic_shape(x)[mask])

                    mask = ~np.all(res== 0, axis=-1)
                    results.append(system_shape(res, x.shape[-2]))
                    # results.append(system_shape(super().transform(atomic_shape(x)), x.shape[-2]))

        if made_list:
            results = results[0]
        if was_tuple:
            return results, y
        else:
            return results

    def fit(self, X, y=None):
        if self.is_fit:
            return self
        else:
            self.is_fit = True
            if isinstance(X, tuple):
                X = X[0]

            if isinstance(X, list):
                super_X = {}
                for x in X:
                    for spec in x:
                        if not spec in super_X:
                            super_X[spec] = []
                        super_X[spec].append(atomic_shape(x[spec]))
                for spec in super_X:
                    super_X[spec] = np.concatenate(super_X[spec])
            else:
                super_X = X

            if isinstance(super_X, dict):
                self._spec_dict = {}
                for spec in super_X:
                    self._spec_dict[spec] =\
                     type(self)(*self._initargs,
                      **self.get_kwargs())
                    self._spec_dict[spec].__dict__.update(self.get_params())
                    # Due to padding some rows might be zero, exclude those during fit:
                    mask = ~np.all(atomic_shape(super_X[spec]) == 0, axis=-1)
                    self._spec_dict[spec].fit(self._before_fit(atomic_shape(super_X[spec])[mask]))
                return self
            else:
                return super().fit(atomic_shape(super_X))

    def get_gradient(self, X, y=None, **fit_params):
        was_tuple = False
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]
            was_tuple = True

        made_list = False
        if not isinstance(X, list):
            X = [X]
            made_list = True

        results = []
        for x in X:
            if isinstance(x, dict):
                results_dict = {}
                for spec in x:
                    results_dict[spec] = self._spec_dict[spec].get_gradient(x[spec])
                results.append(results_dict)
            else:
                results.append(system_shape(self._gradient_function(atomic_shape(x)), x.shape[-2]))

        if made_list:
            results = results[0]
        if was_tuple:
            return results, y
        else:
            return results

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def forward(self, X):
        if self.is_torch:
            return self.transform(X, wrap_torch = False)
        else:
            raise Exception("Must call to_torch before using it as PyTorch Module")

    def to_torch(self):
        TorchModule.__init__(self)
        self.is_torch = True
        if hasattr(self,'_spec_dict'):
            for spec in self._spec_dict:
                self._spec_dict[spec].is_torch = True

class GroupedVarianceThreshold(GroupedTransformer, VarianceThreshold, TorchModule):
    def __init__(self, threshold=0.0):
        """ GroupedTransformer version of sklearn VarianceThreshold.
            See their documentation for more information
        """
        TorchModule.__init__(self)
        self._before_fit = identity  # lambdas can't be pickled
        self._initargs = []
        self.treshold = threshold
        super().__init__(**self.get_kwargs())

    def get_kwargs(self):
        return dict(threshold=self.treshold)

    def _gradient_function(self, X):
        X_shape = X.shape
        if not X.ndim == 2:
            X = X.reshape(-1, X.shape[-1])

        support = self.get_support()
        X_grad = np.zeros([len(X), len(support)])
        X_grad[:, support] = X
        return X_grad.reshape(*X_shape[:-1], X_grad.shape[-1])

    def torch_transform(self, X):
        X_shape = X.size()
        if not len(X_shape) == 2:
            X = X.view(-1,X_shape[-1])
        support = torch.from_numpy(self.get_support()).bool()
        return X[:, support]

class GroupedPCA(GroupedTransformer, PCA, TorchModule):
    def __init__(self,
                 n_components=None,
                 copy=True,
                 whiten=False,
                 svd_solver='auto',
                 tol=0.0,
                 iterated_power='auto',
                 random_state=None):
        """ GroupedTransformer version of sklearn principal component analysis.
            See their documentation for more information
        """

        TorchModule.__init__(self)
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.iterated_power = iterated_power
        self.random_state = random_state

        self._before_fit = StandardScaler().fit_transform
        self._initargs = []
        super().__init__(**self.get_kwargs())

    def get_kwargs(self):
        return dict(n_components=self.n_components,
                    copy=self.copy,
                    whiten=self.whiten,
                    svd_solver=self.svd_solver,
                    tol=self.tol,
                    iterated_power=self.iterated_power,
                    random_state=self.random_state)

    def fit(self, *args, **kwargs):
        if self.n_components == 1:
            return self
        else:
            return super().fit(*args, **kwargs)

    def transform(self, X, y=None, **fit_params):
        if self.n_components == 1:
            return X
        else:
            return super().transform(X, y, **fit_params)

    def get_gradient(self, X, y=None, **fit_params):
        if self.n_components == 1:
            return X
        else:
            return super().get_gradient(X, y, **fit_params)

    def _gradient_function(self, X):
        X_shape = X.shape
        if not X.ndim == 2:
            X = X.reshape(-1, X.shape[-1])
        X_grad = X.dot(self.components_)
        return X_grad.reshape(*X_shape[:-1], X_grad.shape[-1])

    def torch_transform(self, X):
        if self.n_components == 1:
            return X
        else:
            return torch.matmul(X,torch.from_numpy(self.components_).T)


class GroupedStandardScaler(GroupedTransformer, StandardScaler, TorchModule):
    def __init__(self, threshold=0.0):
        """ GroupedTransformer version of sklearn StandardScaler.
            See their documentation for more information
        """
        TorchModule.__init__(self)
        self._before_fit = identity  # lambdas can't be pickled
        self._initargs = []
        self._initkwargs = {}
        super().__init__()

    def get_kwargs(self):
        return {}

    def _gradient_function(self, X):
        X_shape = X.shape
        if not X.ndim == 2:
            X = X.reshape(-1, X.shape[-1])
        X = X / np.sqrt(self.var_).reshape(1, -1)
        return X.reshape(*X_shape[:-1], X.shape[-1])


    def torch_transform(self, X):
        X_shape = X.size()
        if not len(X_shape) == 2:
            X = X.view(-1,X_shape[-1])
        X = (X - torch.from_numpy(self.mean_))/torch.sqrt(torch.from_numpy(self.var_))
        return X

def identity(x):
    return x
