""" Module that implements a Behler-Parinello type neural network
"""

import numpy as np
import torch
from sklearn.base import BaseEstimator
from .pipeline import load_pipeline  #Keep for backwards compatibility of API


class NetworkEstimator(BaseEstimator):

    allows_threading = False

    def __init__(self,
                 n_nodes,
                 n_layers,
                 b,
                 alpha=0.01,
                 max_steps=20001,
                 valid_size=0.2,
                 batch_size=0,
                 activation='sigmoid',
                 **kwargs):
        """ Estimator (scikit-learn) wrapper for the PyTorch based EnergyNetwork class which
        implements a Behler-Parinello type neural network
        """
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.b = b
        self.alpha = alpha
        self.max_steps = max_steps
        self.valid_size = valid_size
        self.path = None
        self._network = None
        self.batch_size = batch_size
        self.activation = activation
        self.verbose = False
        self.fitted = False

    def get_params(self, *args, **kwargs):
        return {
            'n_nodes': self.n_nodes,
            'n_layers': self.n_layers,
            'b': self.b,
            'alpha': self.alpha,
            'max_steps': self.max_steps,
            'valid_size': self.valid_size,
            'batch_size': self.batch_size,
            'activation': self.activation,
        }

    def build_network(self):
        print('building network')
        self._network = EnergyNetwork(n_layers=self.n_layers, n_nodes=self.n_nodes, activation=self.activation)
        if not self.path is None:
            self._network.restore_model(self.path)

    def fit(self, X, y=None, *args, **kwargs):
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]

        if not isinstance(X, list):
            X = [X]

        if not isinstance(y, np.ndarray) and not y:
            y = [np.zeros(len(list(x.values())[0])) for x in X]

        # X = X[0]

        if not self._network:
            self.build_network()
        self._network.train(X[0],
                            y[0],
                            step_size=self.alpha,
                            max_steps=self.max_steps,
                            b_=self.b,
                            train_valid_split=1 - self.valid_size,
                            batch_size=self.batch_size)
        self.fitted = True

    def predict(self, X, *args, **kwargs):

        if self._network is None:
            self.build_network()

        if isinstance(X, tuple):
            X = X[0]
        if not isinstance(X, list):
            X = [X]

        predictions = self._network.predict(X[0])
        return predictions

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

        scores = []
        if not isinstance(X, list):
            X = [X]
            y = [y]

        for X_, y_ in zip(X, y):
            scores.append(metric_function(self.predict(X_) - y_))

        return -np.mean(scores)

    def load_network(self, path):
        self.path = path


def train_net(net, dataloader, dataloader_val=None, max_steps=10000, check_point_every=10, lr=1e-3, weight_decay=1e-7):
    # net.train()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    MIN_RATE = 1e-7
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           verbose=True,
                                                           patience=10,
                                                           min_lr=MIN_RATE)

    max_epochs = max_steps
    for epoch in range(max_epochs):
        logs = {}
        epoch_loss = 0
        for data in dataloader:
            rho, energy = data
            result = net(rho)
            loss = loss_fn(result, energy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        logs['log loss'] = np.sqrt(epoch_loss / len(dataloader))
        for i, param_group in enumerate(optimizer.param_groups):
            logs['lr_{}'.format(i)] = float(param_group['lr'])

        if logs['lr_0'] <= MIN_RATE:
            return net

        if epoch % check_point_every == 0:
            scheduler.step(epoch_loss / len(dataloader))
            val_loss = 0
            if dataloader_val is not None:
                for data in dataloader_val:
                    rho, energy = data
                    result = net(rho)
                    loss = loss_fn(result, energy)
                    val_loss += loss.item()
                logs['val loss'] = np.sqrt(val_loss / len(dataloader_val))
            else:
                logs['val loss'] = 0
            print('Epoch {} ||'.format(epoch), ' Training loss : {:.6f}'.format(logs['log loss']),
                  ' Validation loss : {:.6f}'.format(logs['val loss']), ' Learning rate: {}'.format(logs['lr_0']))
    return net


class Dataset(object):
    def __init__(self, rho, energies):
        self.rho = rho
        self.energies = energies.reshape(-1, 1)

    def __getitem__(self, index):

        rho = {}
        for species in self.rho:
            rho[species] = self.rho[species][index]

        energy = self.energies[index]

        return (rho, energy)

    def __len__(self):
        return len(self.energies)


class EnergyNetwork(torch.nn.Module):
    def __init__(self, n_nodes, n_layers, activation):
        super(EnergyNetwork, self).__init__()
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        if hasattr(torch.nn, activation):
            self.activation = getattr(torch.nn, activation)()
        else:
            print('Activation unknown, defaulting to GELU')
            self.activation = torch.nn.GELU()

    def train(self, X, y, step_size=0.01, max_steps=50001, b_=0, verbose=True, train_valid_split=0.8, batch_size=0):

        if not hasattr(self, 'species_nets'):
            species_nets = {}
            for spec in X:
                species_nets[spec] = torch.nn.Sequential(
                    *([torch.nn.Linear(X[spec].shape[-1], self.n_nodes)] +\
                    (self.n_layers-1)* [self.activation,torch.nn.Linear(self.n_nodes, self.n_nodes)] +\
                    [self.activation, torch.nn.Linear(self.n_nodes,1)])
                )
            self.species_nets = torch.nn.ModuleDict(species_nets)

        if train_valid_split < 1.0:
            indices = np.arange(len(y))
            np.random.shuffle(indices)
            ti = int(len(indices) * train_valid_split)
            train_idx = indices[:ti]
            val_idx = indices[ti:]
            dataset_val = Dataset({spec: X[spec][val_idx] for spec in X}, y[val_idx])
            dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=False)
        else:
            train_idx = np.arange(len(y))
            dataloader_val = None

        dataset_train = Dataset({spec: X[spec][train_idx] for spec in X}, y[train_idx])
        if batch_size == 0:
            batch_size = len(dataset_train)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        train_net(self, dataloader_train, dataloader_val, max_steps=max_steps, lr=step_size, weight_decay=b_)

    def predict(self, X):
        data_len = 0
        for spec in X:
            data_len = max([data_len, len(X[spec])])

        y = np.zeros(data_len)
        dataset = Dataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=data_len, shuffle=False)

        for data in dataloader:
            rho, energy = data
            result = self.forward(rho)
        return [result.detach().numpy()]

    def forward(self, input):
        output = 0
        for spec in input:
            output += torch.sum(self.species_nets[spec](input[spec]), dim=-2)

        return output


Energy_Network = EnergyNetwork  # Needed to unpickle old models
