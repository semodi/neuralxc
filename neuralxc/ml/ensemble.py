from sklearn.base import BaseEstimator
import numpy as np

class EnsembleEstimator(BaseEstimator):

    def __init__(self, estimators, operation = 'sum'):

        allows_threading = False
        self.estimators = estimators
        if hasattr(np, operation):
            self.operation = getattr(np, operation)
        else:
            raise Exception('Operation must be available in numpy, numpy.' + \
                operation +' unknown')
        self.path = None
        self.models_loaded = False

    def load_network(self, path):
        for idx, estimator in enumerate(self.estimators):
            self.estimators[idx].load_network(path + '_e{}'.format(idx))


    def _make_serializable(self, path):

        container = []
        for i, estimator in enumerate(self.estimators):
            container.append(estimator._make_serializable(path + '_e{}'.format(i)))

        return container

    def _restore_after_pickling(self, container):

        assert len(container) == len(self.estimators)
        for c, estimator in zip(container, self.estimators):
            estimator._restore_after_pickling(c)

class ChainedEstimator(EnsembleEstimator):
    allows_threading = False

    def fit(self, X, y=None, **fit_kwargs):
        for estimator in self.estimators[:-1]:
            X = estimator.predict(X, partial=True)

        return self.estimators[-1].fit(X, y, **fit_kwargs)

    def predict(self, X, *args, **kwargs):
        for estimator in self.estimators[:-1]:
            X = estimator.predict(X, *args, partial=True)

        return self.estimators[-1].predict(X, *args, **kwargs)

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
