.. neuralxc documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NeuralXC
=========================================================

.. image:: _static/neuralxc.png
Implementation of a machine learned density functional as presented in `Machine learning accurate exchange and correlation functionals of the electronic density. Nat Commun 11, 3509 (2020) <https://www.nature.com/articles/s41467-020-17265-7>`_

NeuralXC only includes routines to fit and test neural network based density functionals.
To use these functionals for self-consistent calculations within electronic structure codes please refer to `Libnxc <https://github.com/semodi/libnxc/>`_.

The basic premise of NeuralXC can be summarized as follows:

1. The electron density on a real space grid is projected onto a set of atom-centered atomic orbitals
2. The projection coefficients are symmetrized to ensure that systems that only differ by a global rotation have the same energy
3. The symmetrized coefficients are fed through a neural network architecture that is invariant under atom permutations, similar to Behler-Parrinello networks.
4. The output of this neural network is the exchange-correlation (XC) energy (correction) for a given system. The XC-potential is then obtained by backpropagating through steps 1-3.

The very nature of this approach lends itself to a modular implementation. Hence, we have separated NeuralXC into three main
modules, **Projector**, **Symmetrizer**, and **ML**, each of which can be individually customized.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation.rst
   quickstart.rst
   projector.rst
   symmetrizer.rst
   input.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
