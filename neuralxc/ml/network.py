""" Module that implements a Behler-Parinello type neural network
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from matplotlib import pyplot as plt
import math
import pickle
from collections import namedtuple
import h5py
import json
from ase.io import read
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.pipeline import Pipeline
import pickle
import shutil
Dataset = namedtuple("Dataset", "data species")

class NXCPipeline(Pipeline):

    def __init__(self, steps, basis_instructions, symmetrize_instructions):

        self._basis_instructions = basis_instructions
        self._symmetrize_instructions = symmetrize_instructions
        super().__init__(steps)

    def get_symmetrize_instructions(self):
        return self._symmetrize_instructions

    def get_basis_instructions(self):
        return self._basis_instructions

    def _validate_steps(self):
        super()._validate_steps
        names, estimators = zip(*self.steps)


        for t in estimators:
            if t is None:
                continue
            if (not hasattr(t, "get_gradient")):
                raise TypeError("All steps should  "
                                "implement get_gradient."
                                " '%s' (type %s) doesn't" % (t, type(t)))

    @if_delegate_has_method(delegate='_final_estimator')
    def get_gradient(self, X):
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)

        for name, transform in self.steps[::-1]:
            if transform is not None:
                Xt = transform.get_gradient(Xt)

        return Xt

    def start_at(self, step):
        return NXCPipeline(self.steps[step:],
        basis_instructions = self._basis_instructions,
        symmetrize_instructions = self._symmetrize_instructions)

    def save(self, path, override = False):

        if os.path.isdir(path):
            if not override:
                raise Exception('Model already exists, set override = True')
            else:
                shutil.rmtree(path)
                os.mkdir(path)
        else:
            os.mkdir(path)

        network = self.steps[-1][-1]._network
        self.steps[-1][-1]._network = None
        pickle.dump(self, open(os.path.join(path,'pipeline.pckl'),'wb'))
        network.save_model(os.path.join(path,'network'))
        self.steps[-1][-1]._network = network

def load_pipeline(path):
    pipeline = pickle.load(open(os.path.join(path,'pipeline.pckl'),'rb'))
    pipeline.steps[-1][-1].path = os.path.join(path,'network')
    return pipeline

class NetworkEstimator(BaseEstimator):

    def __init__(self, n_nodes, n_layers, b, alpha=0.01,
        max_steps = 20001, test_size = 0.2, valid_size = 0.2,
        random_seed = None) :
        self._n_nodes = n_nodes
        self._n_layers = n_layers
        self._b = b
        self.alpha = alpha
        self.max_steps = max_steps
        self._test_size = test_size
        self._valid_size = valid_size
        self._random_seed = random_seed
        self.path = None
        self._network = None

    def get_params(self, *args, **kwargs):
        return {'n_nodes' : self._n_nodes,
                'n_layers': self._n_layers,
                'b': self._b,
                'alpha': self.alpha,
                'max_steps': self.max_steps,
                'test_size' : self._test_size,
                'valid_size': self._valid_size,
                'random_seed': self._random_seed}

    def build_network(self, X, y=None):
        if isinstance(X, tuple):
            y = X[1]
            X = X[0]

        if not isinstance(X, list):
            X = [X]

        if not isinstance(y, np.ndarray) and not y:
            y = [np.zeros(len(list(x.values())[0])) for x in X]

        subnets = []
        for feat, tar in zip(X,y):
            nets = []
            for spec in feat:
                for j in range(feat[spec].shape[1]):
                    nets.append(Subnet())
                    nets[-1].layers = [self._n_nodes] * self._n_layers
                    nets[-1].activations = [nets[-1].activations[0]] * self._n_layers
                    nets[-1].add_dataset(Dataset(feat[spec][:,j:j+1], spec.lower()),
                        tar, test_size = self._test_size)
            subnets.append(nets)

        self._network = Energy_Network(subnets, scale_again = False)
        if not self.path is None:
            self._network.restore_model(self.path)

    def fit(self,X ,y = None, *args, **kwargs):

        #TODO: Currently does not allow to continue training
        self.build_network(X, y)

        self._network.train(step_size=self.alpha, max_steps=self.max_steps ,
            b_=self._b, train_valid_split=1-self._valid_size,
            random_seed=self._random_seed)


    def get_gradient(self, X, *args, **kwargs):

        if self._network is None:
            self.build_network(X)

        made_list = False

        if isinstance(X, tuple):
            X = X[0]
        if not isinstance(X, list):
            X = [X]
            made_list = True

        X_list = X
        predictions = [{}]*len(X_list)

        for sys_idx, X in enumerate(X_list):
            for spec in X:
                feat = X[spec]
                if feat.ndim == 3:
                    old_shape = feat.shape
                    feat = feat.reshape(-1, feat.shape[-1])

                predictions[sys_idx][spec] = self._network.predict(feat,
                                                    species= spec.lower(),
                                                    *args, return_gradient = True,
                                                    **kwargs)[1].reshape(*old_shape)

        if made_list:
            predictions = predictions[0]
        return predictions


    def predict(self, X, *args, **kwargs):

        if self._network is None:
            self.build_network(X)

        made_list = False
        if isinstance(X, tuple):
            X = X[0]

        if not isinstance(X, list):
            X = [X]
            made_list = True

        X_list = X
        predictions = []

        for X in X_list:
            prediction = 0

            for spec in X:
                feat = X[spec]
                n_sys = len(feat)
                if feat.ndim == 3:
                    feat = feat.reshape(-1, feat.shape[-1])

                prediction += np.sum(self._network.predict(feat,
                                                    species= spec.lower(),
                                                    *args,
                                                    **kwargs).reshape(n_sys,-1), axis = -1)

            predictions.append(prediction)

        if made_list:
            predictions = predictions[0]
        return predictions


    def score(self, X, y = None, metric = 'mae'):

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

        for X_,y_ in zip(X,y):
            scores.append(metric_function(self.predict(X_) - y_))

        return np.mean(scores)

class Energy_Network():
    """ Machine learned correcting functional (MLCF) for energies

        Parameters
        ----------

            subnets: list of Subnetwork
                each subnetwork belongs to a single atom inside the system
                and computes the atomic contributio to the total energy
    """
    def __init__(self, subnets, scale_again = True):


        if not isinstance(subnets, list):
            self.subnets = [subnets]
        else:
            self.subnets = subnets


        self.model_loaded = False
        self.rand_state = np.random.get_state()
        self.graph = None
        self.target_mean = 0
        self.target_std = 1
        self.sess = None
        self.graph = None
        self.initialized = False
        self.optimizer = None
        self.checkpoint_path = None
        self.masks = {}
        self.species_nets = {}
        self.species_nets_names = {}
        self.species_gradients_names = {}
        if scale_again:
            scale_together(subnets)

    # ========= Network operations ============ #

    def __add__(self, other):
        if isinstance(other, Subnet):
            if not len(self.subnets) == 1:
                raise Exception(" + operator only valid if only one training set contained")
            else:
                self.subnets[0] += [other]
        else:
            raise Exception("Datatypes not compatible")

        return self

    def __mod__(self, other):
        if isinstance(other, Subnet):
            self.subnets += [[other]]
        elif isinstance(other, Energy_Network):
            self.subnets += other.subnets
        else:
            raise Exception("Datatypes not compatible")

        return self

    def reset(self):
        self.sess = None
        self.graph = None
        self.initialized = False
        self.optimizer = None
        self.checkpoint_path = None


    def construct_network(self):
        """ Builds the tensorflow graph from subnets
        """

        cnt = 0
        logits = []
        for subnet in self.subnets:
            if isinstance(subnet,list):
                sublist = []
                for s in subnet:
                    sublist.append(s.get_logits(cnt)[0])
                    cnt += 1
                logits.append(sublist)
            else:
                logits.append(subnet.get_logits(cnt)[0])
                cnt += 1

        return logits

    def get_feed(self, which = 'train', train_valid_split = 0.8, seed = 42):
        """ Return a dictionary that can be used as a feed_dict in tensorflow

        Parameters
        -----------
            which: {'train',test'}
                which part of the dataset is used
            train_valid_split: float
                ratio of train and validation set size
            seed: int
                seed parameter for the random shuffle algorithm, make

        Returns
        --------
            (dictionary, dictionary)
                either (training feed dictionary, validation feed dict.)
                or (testing feed dictionary, None)
        """
        train_feed_dict = {}
        valid_feed_dict = {}
        test_feed_dict = {}

        for subnet in self.subnets:
            if isinstance(subnet,list):
                for s in subnet:
                    train_feed_dict.update(s.get_feed('train', train_valid_split, seed))
                    valid_feed_dict.update(s.get_feed('valid', train_valid_split, seed))
                    test_feed_dict.update(s.get_feed('test', train_valid_split, seed))
            else:
                train_feed_dict.update(subnet.get_feed('train', train_valid_split, seed))
                valid_feed_dict.update(subnet.get_feed('valid', train_valid_split, seed))
                test_feed_dict.update(subnet.get_feed('test', seed = seed))

        if which == 'train':
            return train_feed_dict, valid_feed_dict
        elif which == 'test':
            return test_feed_dict, None


    def get_cost(self):
        """ Build the tensorflow node that defines the cost function

        Returns
        -------
            cost_list: [tensorflow.placeholder]
                list of costs for subnets. subnets
                whose outputs are added together share cost functions
        """
        cost_list = []

        for subnet in self.subnets:
            if isinstance(subnet,list):
                cost = 0
                y_ = self.graph.get_tensor_by_name(subnet[0].y_name)
                log = 0
                for s in subnet:
                    log += self.graph.get_tensor_by_name(s.logits_name)
                cost += tf.reduce_mean(tf.reduce_mean(tf.square(y_-log),0))
            else:
                log = self.graph.get_tensor_by_name(subnet.logits_name)
                y_ = self.graph.get_tensor_by_name(subnet.y_name)
                cost = tf.reduce_mean(tf.reduce_mean(tf.square(y_-log),0))
            cost_list.append(cost)

        return cost_list




    def train(self,
              step_size=0.01,
              max_steps=50001,
              b_=0,
              verbose=True,
              optimizer=None,
              adaptive_rate=False,
              multiplier = 1.0,
              train_valid_split = 0.8,
              random_seed = None):

        """ Train the master neural network

            Parameters
            ----------
                step_size: float
                    step size for gradient descent
                max_steps: int
                    number of training epochs
                b: float
                    regularization parameter
                verbose: boolean
                    print cost for intermediate training epochs
                optimizer: tf.nn.GradientDescentOptimizer,tf.nn.AdamOptimizer, ...
                adaptive_rate: boolean
                    wether to adjust step_size if cost increases
                                not recommended for AdamOptimizer
                multiplier: list of float
                    multiplier that allow to give datasets more
                    weight than others

            Returns
            --------
            None
        """


        self.model_loaded = True
        if self.graph is None:
            self.graph = tf.Graph()
            build_graph = True
        else:
            build_graph = False

        with self.graph.as_default():

            if self.sess == None:
                sess = tf.Session()
                self.sess = sess
            else:
                sess = self.sess


            if not random_seed is None:
                tf.random.set_random_seed(random_seed)
            # Get number of distinct subnet species
            species = {}
            for net in self.subnets:
                if isinstance(net,list):
                    for net in net:
                        for l,_ in enumerate(net.layers):
                            name = net.species
                            species[name] = 1
                else:
                    for l,_ in enumerate(net.layers):
                        name = net.species
                        species[name] = 1
            n_species = len(species)


            # Build all the required tensors
            b = {}
            # if build_graph:
            #     self.construct_network()
            #     for s in species:
            #         b[s] = tf.placeholder(tf.float32,name = '{}/b'.format(s))
            # else:
            #     for s in species:
            #         b[s] = self.graph.get_tensor_by_name('{}/b:0'.format(s))
            #

            self.construct_network()
            for s in species:
                b[s] = tf.placeholder(tf.float32,name = '{}/b'.format(s))
            cost_list = self.get_cost()
            train_feed_dict, valid_feed_dict = self.get_feed('train',
             train_valid_split=train_valid_split)
            cost = 0
            if not isinstance(multiplier, list):
                multiplier = [1.0]*len(cost_list)
            print('multipliers: {}'.format(multiplier))
            for c, m in zip(cost_list, multiplier):
                cost += c*m

            # L2-loss
            loss = 0
            with tf.variable_scope("", reuse=True):
                for net in self.subnets:
                    if isinstance(net,list):
                        for net in net:
                            for l, layer in enumerate(net.layers):
                                name = net.species
                                loss += tf.nn.l2_loss(tf.get_variable("{}/W{}".format(name, l+1))) * \
                                        b[name]/layer
                    else:
                        for l, layer in enumerate(net.layers):
                            name = net.species
                            loss += tf.nn.l2_loss(tf.get_variable("{}/W{}".format(name, l+1))) * \
                                b[name]/layer

            cost += loss

            for i, s in enumerate(species):
                train_feed_dict['{}/b:0'.format(s)] = b_[i]
                valid_feed_dict['{}/b:0'.format(s)] = 0


            if self.optimizer == None:
                if optimizer == None:
                    self.optimizer = tf.train.AdamOptimizer(learning_rate = step_size)
                else:
                    self.optimizer = optimizer

            train_step = self.optimizer.minimize(cost)

            # Workaround to load the AdamOptimizer variables
            if not self.checkpoint_path == None:
                saver = tf.train.Saver()
                saver.restore(self.sess,self.checkpoint_path)
                self.checkpoint_path = None

            initialize_uninitialized(self.sess)

            self.initialized = True

            train_writer = tf.summary.FileWriter('./log/',
                                      self.graph)
            old_cost = 1e8

            for _ in range(0,max_steps):

                sess.run(train_step,feed_dict=train_feed_dict)

                if _%int(max_steps/100) == 0 and adaptive_rate == True:
                    new_cost = sess.run(tf.sqrt(cost),
                        feed_dict=train_feed_dict)

                    if new_cost > old_cost:
                        step_size /= 2
                        print('Step size decreased to {}'.format(step_size))
                        train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)
                    old_cost = new_cost

                if _%int(max_steps/10) == 0 and verbose:
                    print('Step: ' + str(_))
                    print('Training set loss:')
                    if len(cost_list) > 1:
                        for i, c in enumerate(cost_list):
                            print('{}: {}'.format(i,sess.run(tf.sqrt(c),feed_dict=train_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost-loss),feed_dict=train_feed_dict)))
                    print('Validation set loss:')
                    if len(cost_list) > 1:
                        for i, c in enumerate(cost_list):
                            print('{}: {}'.format(i,sess.run(tf.sqrt(c),feed_dict=valid_feed_dict)))
                    print('Total: {}'.format(sess.run(tf.sqrt(cost),feed_dict=valid_feed_dict)))
                    print('--------------------')
                    print('L2-loss: {}'.format(sess.run(loss,feed_dict=train_feed_dict)))

    def predict(self, features, species, use_masks = False, return_gradient = False):
        """ Get predicted energies

        Parameters
        ----------
        features: np.ndarray
            input features
        species: str
            predict atomic contribution to energy for this species
        use_masks: bool
             whether masks should be applied to the provided features
        return_gradient: bool
            instead of returning energies, return gradient of network
            w.r.t. input features

        Returns
        -------
        np.ndarray
            predicted energies or gradient

        """
        species = species.lower()
        if features.ndim == 2:
            features = features.reshape(-1,1,features.shape[1])
        else:
            raise Exception('features.ndim != 2')
        if use_masks:
            features = features[:,:,self.masks[species]]

        ds = Dataset(features, species)
        targets = np.zeros(features.shape[0])

        if not species in self.species_nets:
            self.species_nets[species] = Subnet()
            found = False

            snet = self.species_nets[species]
            for s in self.subnets:
                if found == True:
                    break
                if isinstance(s,list):
                    for s2 in s:
                        if s2.species == ds.species:
                            snet.layers = s2.layers
                            snet.targets = s2.targets
                            snet.activations = s2.activations
                            found = True
                            break
                else:
                    if s.species == ds.species:
                        snet.layers = s.layers
                        snet.targets = s.targets
                        snet.activations = s.activations
                        break

        snet = self.species_nets[species]
        snet.add_dataset(ds, targets, test_size=0.0, scale = False)
        if not self.model_loaded:
            raise Exception('Model not loaded!')
        else:
            with self.graph.as_default():
                if species in self.species_nets_names:
                    logits = self.graph.get_tensor_by_name(self.species_nets_names[species])
                    gradients = self.graph.get_tensor_by_name(self.species_gradients_names[species])
                else:
                    logits, x, _ = snet.get_logits(1)
                    gradients = tf.gradients(logits, x)[0].values
                    self.species_nets_names[species] = logits.name
                    self.species_gradients_names[species] = gradients.name
                sess = self.sess
                energies = sess.run(logits, feed_dict=snet.get_feed(which='train',
                     train_valid_split=1.0))
                if return_gradient:
                    grad = sess.run(gradients, feed_dict=snet.get_feed(which='train',
                     train_valid_split=1.0))[0]
                    # grad = grad/np.sqrt(snet.scaler.var_).reshape(1,-1)
                    energies = (energies, grad)

                return energies

    def get_weights(self, species, subnet):
        W = []
        b = []
        with self.graph.as_default():
            with tf.variable_scope("", reuse=True):
                for l, layer in enumerate(subnet.layers + ['']):
                    W.append(self.sess.run(tf.get_variable("{}/W{}".format(species, l + 1))))
                    b.append(self.sess.run(tf.get_variable("{}/b{}".format(species, l + 1))))
        return W, b
    def get_energies(self, summarize = True, which = 'train'):
        """ Uses trained model on training or test sets

        Parameters
        -----------
        which: str
            {'train','test'} which set logits are computed for

        Returns
        --------
         list of numpy.ndarray
            resulting energies grouped by independent subnet datasets
        """

        if not self.model_loaded:
            raise Exception('Model not loaded!')
        else:
            with self.graph.as_default():

                logits_list = self.construct_network()

                sess = self.sess
                feed_dict, _ = self.get_feed(train_valid_split = 1.0, which = which)

                return_list = []

                for logits in logits_list:
                    if isinstance(logits,list):
                        result = 0
                        for logits in logits:
                            if summarize:
                                result += sess.run(logits,feed_dict=feed_dict)
                            else:
                                return_list.append(sess.run(logits,feed_dict=feed_dict))
                        if summarize:
                            return_list.append(result)
                    else:
                        return_list.append(sess.run(logits,feed_dict=feed_dict))
                return return_list

    def save_species_model(self, path, species):

        if path[-5:] == '.ckpt':
            path = path[:-5]

        snet = self.species_nets[species]
        path = path + '_' + species
        with self.graph.as_default():
                if species in self.species_nets_names:
                    logits = self.graph.get_tensor_by_name(self.species_nets_names[species])
                    gradients = self.graph.get_tensor_by_name(self.species_gradients_names[species])
                else:
                    logits, x, _ = snet.get_logits(1)
                    gradients = tf.gradients(logits, x)[0].values
                    self.species_nets_names[species] = logits.name
                    self.species_gradients_names[species] = gradients.name
                sess = self.sess
                saver = tf.train.Saver()
                saver.save(sess,save_path = path + '.ckpt')

    def save_model(self, path):
        """ Save trained model to path
        """

        if path[-5:] == '.ckpt':
            path = path[:-5]

        with self.graph.as_default():
            sess = self.sess
            saver = tf.train.Saver()
            saver.save(sess,save_path = path + '.ckpt')

    def restore_model(self, path):
        """ Load trained model from path
        """

        if path[-5:] == '.ckpt':
            path = path[:-5]

        self.checkpoint_path = path + '.ckpt'
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session()
            self.construct_network()
            b = tf.placeholder(tf.float32,name = 'b')
            saver = tf.train.Saver()
            saver.restore(sess,path + '.ckpt')
            self.model_loaded = True
            self.sess = sess
            self.graph = g
            self.initialized = True

    def save_all(self, net_dir, override = False):
        """ Saves the model including all subnets and datasets
        using pickle to directory net_dir, if directory exists only
        save if override = True
        """
        try:
            os.mkdir(net_dir)
        except FileExistsError:
            if override:
                print('Overriding...')
                import shutil
                shutil.rmtree(net_dir)
                os.mkdir(net_dir)
            else:
                print('Directory/Network already exists. Network not saved...')
                return None

        # Pickle does not seem to be compatible with tensorflow so just
        # save subnetworks with it
        with open(os.path.join(net_dir,'subnets'),'wb') as file:
            pickle.dump(self.subnets,file)

        to_save = {'mask': self.masks}

        self.save_model(os.path.join(net_dir,'model'))

        pickle.dump(to_save, open(os.path.join(net_dir,'supp'), 'wb'))


    def load_all(self, net_dir):
        """ Loads the model in net_dir including all subnets and datasets
        using pickle
        """

        with open(os.path.join(net_dir,'subnets'),'rb') as file:
            self.subnets = pickle.load(file)

        self.restore_model(os.path.join(net_dir,'model'))
        self.masks = pickle.load(open(os.path.join(net_dir,'supp'), 'rb'))['mask']

class Subnet():
    """ Subnetwork that is associated with one Atom
    """

    seed = 42

    def __init__(self):
        self.species = None
        self.n_copies = 0
        self.rad_param = None
        self.ang_param = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.name = None
        self.constructor = fc_nn_g
        self.logits_name = None
        self.x_name = None
        self.y_name = None
        self.layers = [8] * 3
        self.targets = 1
        self.activations = [tf.nn.sigmoid] * 3

    def __add__(self, other):
        if not isinstance(other,Subnet):
            raise Exception("Incompatible data types")
        else:
            return Energy_Network([[self,other]])

    def __mod__(self, other):
        if not isinstance(other,Subnet):
            raise Exception("Incompatible data types")
        else:
            return Energy_Network([[self],[other]])


    def get_feed(self, which, train_valid_split = 0.8, seed = None):
        """ Return a dictionary that can be used as a feed_dict in tensorflow

        Parameters
        -----------
            which: str,
                {'train', 'valid', 'test'}
                which part of the dataset is used
            train_valid_split: float
                ratio of train and validation set size
            seed: int
                seed parameter for the random shuffle algorithm

        Returns
        --------
            dict
        """
        if seed == None:
            seed = Subnet.seed

        if train_valid_split == 1.0:
            shuffle = False
        else:
            shuffle = True


        if which == 'train' or which == 'valid':

            X_train = np.concatenate([self.X_train[i] for i in range(self.n_copies)],
                axis = 1)

            X_train, X_valid, y_train, y_valid = \
                train_test_split(X_train,self.y_train,
                                 test_size = 1 - train_valid_split,
                                 random_state = seed, shuffle = shuffle)
            X_train, X_valid = reshape_group(X_train, self.n_copies) , \
                               reshape_group(X_valid, self.n_copies)

            if which == 'train':
                return {self.x_name : X_train, self.y_name: y_train}
            else:
                return {self.x_name : X_valid, self.y_name : y_valid}

        elif which == 'test':

            return {self.x_name : self.X_test, self.y_name: self.y_test}



    def get_logits(self, i):
        """ Builds the subnetwork by defining logits and placeholders

        Parameters
        -----------
            i: int
                index to label datasets

        Returns
        ---------
            tensorflow tensors
        """

        with tf.variable_scope(self.name) as scope:
                        try:
                            logits,x,y_ = self.constructor(self, i, np.mean(self.targets), np.std(self.targets))
                        except ValueError:
                            scope.reuse_variables()
                            logits,x,y_ = self.constructor(self, i, np.mean(self.targets), np.std(self.targets))

        self.logits_name = logits.name
        self.x_name = x.name
        self.y_name = y_.name
        return logits, x, y_

    def save(self, path):
        """ Use pickle to save the subnet to path
        """

        with open(path,'wb') as file:
            pickle.dump(self,file)

    def load(self, path):
        """ Load subnet from path
        """

        with open(path, 'rb') as file:
            self = pickle.load(file)

    def add_dataset(self, dataset, targets,
        test_size = 0.2, target_filter = None, scale = False):
        """ Adds dataset to the subnetwork.

            Parameters
            -----------
                dataset: dataset
                    contains datasets that will be associated with subnetwork for training and
                    evaluation
                targets: np.ndarray
                    target values for training and evaluation

            Returns
            --------
                None
        """

        if self.species != None:
            if self.species != dataset.species:
                raise Exception("Dataset species does not equal subnet species")
        else:
            self.species = dataset.species

        if not self.n_copies == 0:
            if self.n_copies != dataset.data.shape[1]:
                raise Exception("New dataset incompatible with contained one.")

        self.n_copies = dataset.data.shape[1]
        self.name = dataset.species


        if not test_size == 0.0:
            X_train, X_test, y_train, y_test = \
                train_test_split(dataset.data, targets,
                    test_size= test_size, random_state = Subnet.seed, shuffle = True)
        else:
            X_train = dataset.data
            y_train = targets
            X_test = np.array(X_train)
            y_test = np.array(y_train)

        if scale:
            if self.scaler == None:
                scaler = StandardScaler()
                # scaler = MinMaxScaler(feature_range=(-2,2))
                scaler.fit(X_train.reshape(-1, X_train.shape[2]))
            else:
                scaler = self.scaler

            X_train = scaler.transform(X_train.reshape(-1,
                X_train.shape[2])).reshape(X_train.shape)
            X_test = scaler.transform(X_test.reshape(-1,
                X_test.shape[2])).reshape(X_test.shape)

            self.scaler = scaler

        self.X_train = X_train.swapaxes(0,1)
        self.X_test = X_test.swapaxes(0,1)
        self.features = X_train.shape[2]

        if y_train.ndim == 1:
            self.y_train = y_train.reshape(-1,1)
            self.y_test = y_test.reshape(-1,1)
        else:
            self.y_train = y_train
            self.y_test = y_test

        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


def load_energy_model(path):
    """ Load energy MLCF

        Parameters
        ---------
            path: str
                directory in which energy MLCF is stored

        Returns
        -------
            Energy_Network
    """
    network = Energy_Network([])
    network.load_all(path)
    return network

def fc_nn(network):
    """Builds a fully connected neural network

    Parameters
    -----------------
        network: Subnet

    Returns
    --------
        logits: tensorflow tensor
            output layer of neural network
        x: tensorflow placeholder
            input layer
        y:  tensorflow placeholder
            placeholder for target values
        """


    features = network.features
    layers = network.layers
    targets = network.targets
    activations = network.activations
    namescope = network.name

    n = len(layers)

    W = []
    b = []
    hidden = []

    x = tf.placeholder(tf.float32,[None,features])
    y_ = tf.placeholder(tf.float32,[None,targets])



    W.append(tf.get_variable(initializer = tf.truncated_normal_initializer(),shape = [features,layers[0]],name='W1'))
    b.append(tf.get_variable(initializer = tf.constant_initializer(0),shape = [layers[0]],name='b1'))
    hidden.append(activations[0](tf.matmul(x,W[0])+b[0]))

    for l in range(1,n):
        W.append(tf.get_variable(initializer = tf.truncated_normal_initializer(),shape = [layers[l-1],layers[l]],name='W' + str(l+1)))
        b.append(tf.get_variable(initializer = tf.constant_initializer(0),shape = [layers[l]],name='b' + str(l+1)))
        hidden.append(activations[l](tf.matmul(hidden[l-1],W[l])+b[l]))

    W.append(tf.get_variable(initializer = tf.truncated_normal_initializer, shape= [layers[n-1],targets],name='W' + str(n+1)))
    b.append(tf.get_variable(initializer = tf.constant_initializer(0), shape = [targets],name='b' + str(n+1)))


    logits = tf.matmul(hidden[n-1],W[n])+b[n]

    return logits,x,y_


def fc_nn_g(network, i, mean = 0, std = 1):
    """Builds a fully connected neural network that consists of network.n_copies
    copies of a subnetwork

    Parameters
    ----------
        network: Subnet
            subnet object
        i: int
            index to label placeholders (for multiple datasets)
        mean: float
            mean target value
        std: float
            standard deviation of target values

    Returns
    --------
        logits: tensorflow tensor
            output layer of neural network
        x: tensorflow placeholder
            input layer
        y:  tensorflow placeholder
            placeholder for target values
        """

    features = network.features
    layers = network.layers
    targets = network.targets
    activations = network.activations
    namescope = network.name
    n_copies = network.n_copies
    mean = mean/network.n_copies
    std = std/network.n_copies

    n = len(layers)

    W = []
    b = []
    hidden = []
    x = tf.placeholder(tf.float32,[n_copies,None,features],'x' + str(i))
    y_ = tf.placeholder(tf.float32,[None, targets], 'y_' + str(i))



    W.append(tf.get_variable(initializer = tf.truncated_normal_initializer(),shape = [features,layers[0]],name='W1'))
    b.append(tf.get_variable(initializer = tf.constant_initializer(0),shape = [layers[0]],name='b1'))


    for l in range(1,n):
        W.append(tf.get_variable(initializer = tf.truncated_normal_initializer(),shape = [layers[l-1],layers[l]],name='W' + str(l+1)))
        b.append(tf.get_variable(initializer = tf.constant_initializer(0),shape = [layers[l]],name='b' + str(l+1)))


    W.append(tf.get_variable(initializer = tf.random_normal_initializer(0,std), shape= [layers[n-1],targets],name='W' + str(n+1)))
    b.append(tf.get_variable(initializer = tf.constant_initializer(mean), shape = [targets],name='b' + str(n+1)))


    for n_g in range(n_copies):
        # hidden.append(activations[0](tf.matmul(tf.gather(x,n_g),W[0])/features*10 + b[0]))
        hidden.append(activations[0](tf.matmul(tf.gather(x,n_g),W[0]) + b[0]))
        for l in range(0,n-1):
            # hidden.append(activations[l+1](tf.matmul(hidden[n_g*n+l],W[l+1])/layers[l]*10 + b[l+1]))
            hidden.append(activations[l+1](tf.matmul(hidden[n_g*n+l],W[l+1]) + b[l+1]))

        if n_g == 0:
            logits = tf.matmul(hidden[n_g*n+n-1],W[n])+b[n]
        else:
            logits +=  tf.matmul(hidden[n_g*n+n-1],W[n])+b[n]

    return logits,x,y_

def initialize_uninitialized(sess):
    """ Search graph for uninitialized variables and initialize them
    """
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def reshape_group(x, n):
    """Reshape data from format (n_samples, n_copies * 4)
    into format (n_copies, n_samples, 4) needed by tensorflow
    """

    n0 = x.shape[0]
    n1 = int(x.shape[1]/n)
    x = x.T.reshape(n,n1,n0).swapaxes(1,2)

    return x

def scale_together(all_subnets):
    """ Scale data in given subnets by combining the data ranges
        (Should always be performed if subnets share weights)
    """
    print('scale_together')
    subnet_dict = {}

    def add_to_dict(sn):
        if not sn.species in subnet_dict:
            subnet_dict[sn.species] = []
        subnet_dict[sn.species].append(sn)

    for sn1 in all_subnets:
        if isinstance(sn1, list):
            for sn1 in sn1:
                add_to_dict(sn1)
        else:
            add_to_dict(sn1)
    for species in subnet_dict:
        subnets = subnet_dict[species]
        for s in subnets:
            if not s.species == subnets[0].species:
                raise Exception('Subnets do not contain the same species. Proceeding...')

        if not isinstance(subnets ,list):
            Exception('Input must be a list of subnets')

        all_data = np.concatenate([s.scaler.inverse_transform(s.X_train.reshape(-1, s.features)) for s in subnets])
        # new_scaler = MinMaxScaler(feature_range = (-2,2))
        new_scaler = StandardScaler()
        new_scaler.fit(all_data)
        for i,s in enumerate(subnets):
            subnets[i].X_train = s.scaler.inverse_transform(s.X_train.reshape(-1,s.features)).reshape(s.X_train.shape)
            subnets[i].X_test = s.scaler.inverse_transform(s.X_test.reshape(-1,s.features)).reshape(s.X_test.shape)
            subnets[i].scaler = new_scaler
            subnets[i].X_train = s.scaler.transform(subnets[i].X_train.reshape(-1,s.features)).reshape(s.X_train.shape)
            subnets[i].X_test = s.scaler.transform(subnets[i].X_test.reshape(-1,s.features)).reshape(s.X_test.shape)
