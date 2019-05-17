from sklearn.base import BaseEstimator
import numpy as np

class StackedEstimator(BaseEstimator):

    def __init__(self, estimators, operation = 'sum'):

        self.estimators = estimators
        if hasattr(np, operation):
            self.operation = getattr(np, operation)
        else:
            raise Exception('Operation must be available in numpy, numpy.' + \
                operation +' unknown')
        self.path = None
        self.models_loaded = False

    def fit(self, X, y=None, **fit_kwargs):
        raise NotImplementedError('Cannot directly fit a StackedEstimator')

    def predict(self, X, *args, **kwargs):

        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(X, *args, **kwargs))

        return self.operation(np.array(predictions), axis =0)


    def load_network(self, path):
        for idx, estimator in enumerate(self.estimators):
            self.estimators[idx].load_network(path _ '_e{}'.format(i))


    def _make_serializable(self, path):

        container = []
        for i, estimator in enumerate(self.estimators):
            container.append(networks._make_serializable(path + '_e{}'.format(i)))

        return container

    def _restore_after_pickling(self, container):

        assert len(container) == len(self.estimators)
        for c, estimator in zip(container, self.estimators):
            estimator._restore_after_pickling(c)
