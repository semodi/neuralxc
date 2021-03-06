"""
neuralxc
Implementation of a machine learned density functional
"""

# Add imports here
import warnings
warnings.filterwarnings("ignore")
from .neuralxc import NeuralXC, SiestaNXC, get_nxc_adapter, verify_type, get_V
from . import projector
from . import utils
from . import constants
from . import symmetrizer
from . import ml
from . import base
from . import datastructures
from . import drivers
from . import pyscf
# from . import formatter
#from .projector import *
#from .mlpipeline import *
#from .symmetrizer import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
