from sklearn.base import BaseEstimator
from neuralxc.ml.network import NumpyNetworkEstimator
import numpy as np

class EnsembleEstimator(BaseEstimator):

    def __init__(self, estimators, operation = 'sum'):

        allows_threading = False
        self.estimators = estimators
        if isinstance(operation, str):
            if hasattr(np, operation):
                self.operation = getattr(np, operation)
            else:
                raise Exception('Operation must be available in numpy, numpy.' + \
                    operation +' unknown')
        else:
            self.operation = operation
        self.path = None
        self.models_loaded = False

    def load_network(self, path):
        for idx, estimator in enumerate(self.estimators):
            self.estimators[idx].load_network(path + '_e{}'.format(idx))


    def get_np_estimator(self):
        np_estimators = []
        for est in self.estimators:
            np_estimators.append(est.get_np_estimator())
        return type(self)(np_estimators,self.operation)

    def _make_serializable(self, path):

        container = []
        for i, estimator in enumerate(self.estimators):
            container.append(estimator._make_serializable(path + '_e{}'.format(i)))

        return container

    def _restore_after_pickling(self, container):

        assert len(container) == len(self.estimators)
        for c, estimator in zip(container, self.estimators):
            estimator._restore_after_pickling(c)

    def score(self, X, y=None, metric='mae'):

        if isinstance(X, tuple):
            y = X[1]
            X = X[0]

        if metric == 'mae':
            metric_function = (lambda x: np.mean(np.abs(x)))
        elif metric == 'rmse':
            metric_function = (lambda x: np.sqrt(np.mean(x**2)))
        else:
            raise Exception('Metric unknown or not implemented')


    #         metric_function = (lambda x: np.mean(np.abs(x)))
        scores = []
        if not isinstance(X, list):
            X = [X]
            y = [y]

        for X_, y_ in zip(X, y):
            scores.append(metric_function(self.predict(X_) - y_))

        return -np.mean(scores)

class ChainedEstimator(EnsembleEstimator):
    allows_threading = False

    def fit(self, X, y=None, **fit_kwargs):
        if isinstance(X, tuple):
            y = X[1]

        for estimator in self.estimators[:-1]:
            X = estimator.predict(X, partial=True)

        return self.estimators[-1].fit(X, y, **fit_kwargs)

    def predict(self, X, *args, **kwargs):
        for estimator in self.estimators[:-1]:
            X = estimator.predict(X, *args, partial=True)

        return self.estimators[-1].predict(X, *args, **kwargs)


    def set_params(self, **kwargs):
        self.estimators[-1].set_params(**kwargs)

    def get_params(self, *args, **kwargs):
        return self.estimators[-1].get_params(*args, **kwargs)

    def get_gradient(self, X, *args, **kwargs):

        raise NotImplementedError('ChainedEstimators can currently not be used for DFT calculations' +\
        ' as the get_gradient() method has not been thoroughly tested. Use merge() to convert' +\
        ' this ChainedEstimator into a single NumpyNetworkEstimator')
        gradients = {}

        for estimator in self.estimators:
            grad = estimator.get_gradient(X, *args, **kwargs)
            X = estimator.predict(X, *args, partial=True)
            for species in grad:
                if not species in gradients:
                    gradients[species] = []
                gradients[species].append(grad[species])

        for species in gradients:
            grad = gradients[species][0]
            for g in gradients[species][1:-1]:
                grad = np.einsum('imkj, imjl -> imkl', g, grad)
            grad = np.einsum('imj,imjl -> iml', gradients[species][-1], grad)
            gradients[species] = grad

        return gradients

    def merge(self):

        W = {}
        B = {}
        act = self.estimators[0].activation
        for estimator in self.estimators:
            if not isinstance(estimator, NumpyNetworkEstimator):
                raise Exception('Merging only possible if all estimators are numpy based')

            w = estimator.W
            b = estimator.B
            if not isinstance(estimator.activation,type(act)):
                raise Exception('Activations not consistent across layers.\
                        Merging not supported')
            act = estimator.activation

            for spec in w:
                if not spec in W:
                    W[spec] = []
                    B[spec] = []
                for ws, bs in zip(w[spec],b[spec]):
                    W[spec].append(ws)
                    B[spec].append(bs)
        return NumpyNetworkEstimator(W, B, act)

class StackedEstimator(EnsembleEstimator):
    allows_threading = False

    def fit(self, X, y=None, **fit_kwargs):
        raise NotImplementedError('Cannot directly fit a StackedEstimator')

    def predict(self, X, *args, **kwargs):

        if kwargs.get('partial', False):

            predictions = {}

            for estimator in self.estimators:
                pred = estimator.predict(X, *args, **kwargs)[0]
                assert isinstance(pred, dict)
                for species in pred:
                    if not species in predictions:
                        predictions[species] = []
                    predictions[species].append(pred[species])

            for species in predictions:
                predictions[species] = self.operation(np.array(predictions[species]), axis = 0)
            return [predictions]
        else:
            predictions = []
            for estimator in self.estimators:
                predictions.append(estimator.predict(X, *args, **kwargs))

            return self.operation(np.array(predictions), axis =0)


    def get_gradient(self, X, *args, **kwargs):
        gradients = {}

        for estimator in self.estimators:
            grad = estimator.get_gradient(X, *args, **kwargs)
            for species in grad:
                if not species in gradients:
                    gradients[species] = []
                gradients[species].append(grad[species])

        for species in gradients:
            gradients[species] = self.operation(np.array(gradients[species]), axis = 0)
        return gradients
