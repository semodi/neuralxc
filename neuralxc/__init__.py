"""
neuralxc
Implementation of a machine learned density functional
"""

# Add imports here
import warnings

warnings.filterwarnings("ignore")
PYSCF_FOUND = True
import torch

# try:
#     import pyscf
# except ModuleNotFoundError:
#     PYSCF_FOUND = False
from . import (base, config, constants, datastructures, drivers, ml, projector, pyscf, symmetrizer, utils)
# Handle versioneer
from ._version import get_versions
from .neuralxc import NeuralXC, PySCFNXC

# from .neuralxc import NeuralXC as NeuralXCJIT

versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
