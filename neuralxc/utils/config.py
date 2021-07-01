from collections.abc import MutableMapping, Mapping

import hashlib
import json
from copy import deepcopy


default_config_pyscf = {
    "preprocessor":
    {
        "basis": {"name" : "ccpvdz-jkfit"},
        "projector": "gaussian",
        "grid": "analytical",
        "grad" : 0,
        "operator" : "delta",
        "delta": False,
        "dfit": False
    },
    "engine":
    {
        "application": "pyscf",
        "xc": "PBE",
        "basis" : "def2-TZVP",
        "extension": "chkpt",
    },
    "n_workers" : 1
}

default_config_siesta = {
    "preprocessor":
    {
        "basis": {"n": 3, "l" : 4, "r_o": 2.0},
        "projector": "ortho",
        "grid": "euclidean",
    },
    "engine": {
        "application": "siesta",
        "pseudoloc" : ".",
        "fdf_path" : None,
        "xc": "PBE",
        "basis" : "DZP",
        "fdf_arguments": {"MaxSCFIterations": 50},
        "extension": "RHOXC"
    },
    "n_workers" : 1
}

defaults = {'pyscf' : default_config_pyscf,
            'siesta': default_config_siesta}


def find_projector_type(config):
    pre = config["preprocessor"]
    ptype = pre["projector"]
    if pre["grid"] == "radial":
        ptype += "_radial"
    elif pre["grid"] == "analytical":
        if not config["engine"]["application"] == "pyscf":
            raise ValueError("Analytical projection only supported if application is PySCF")
        ptype = 'pyscf'
    elif pre["grid"] == "euclidean":
        if config["engine"]["application"] == "pyscf":
            raise ValueError("PySCF does not support euclidean grids.")
    else:
        raise ValueError("Grid type must be either euclidean, radial or analytical.")

    config['preprocessor']['projector_type'] = ptype

def fix_basis(config):
    pre = config['preprocessor']
    ptype = pre['projector_type']
    agnostic = False
    basis = {}
    if 'ortho' in ptype:
        if not isinstance(pre['basis'], dict):
            raise ValueError('Dict expected for "basis"')
        if any([isinstance(val, dict) for val in pre['basis'].values()]):
            basis.update(pre['basis'])
        else:
            basis['X'] =  pre['basis']
            agnostic = True
    elif 'pyscf' == ptype or 'gaussian' in ptype:
        if isinstance(pre['basis'], str): #Short-hand notation for PySCF basis sets
            basis['basis'] = {'name': pre['basis']}
            agnostic = False
        elif isinstance(pre['basis'], dict):
            if 'name' in pre['basis']:
                basis['basis'] = pre['basis']
                agnostic = False
            elif 'file' in pre['basis']:
                basis['X'] = {}
                basis['X'].update(pre['basis'])
                basis['X']['basis'] = basis['X'].pop('file')
                agnostic = True

    config._basis.update(basis)
    config._basis['spec_agnostic'] = agnostic
    pre["extension"] = config["engine"].get("extension","chkpt")
    application = config["engine"].get("application","chkpt")
    if application == 'pyscf' and pre['grid'] == 'radial':
        application = 'pyscf_rad'
    pre["application"] = application

class BasisInstructions(MutableMapping):

    def __init__(self, preprocessor, realbasis):
        self.hash = hashlib.md5(json.dumps(preprocessor).encode()).hexdigest()
        self.__dict__.update(preprocessor)
        self.__dict__.update(realbasis)
        self.species = [key for key in realbasis if len(key) < 3]

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __delitem__(self, key):
        del self.__dict__[key]

    def __getitem__(self, k):
        return self.__dict__[k]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def update(self, d):
        self.__dict__.update(d)

class ConfigFile(MutableMapping):

    private_keys = ['_basis', '_complete','_hash']

    def __init__(self, content):

        # if content is file path
        if isinstance(content, str):
            content = json.loads(open(content, 'r').read())

        if not isinstance(content, Mapping):
            raise TypeError('content has to be Mapping')

        default_content = {}
        if 'application' in content:
            default_content = deepcopy(defaults[content['application']])
        elif 'application' in content.get('engine',{}):
            default_content = deepcopy(defaults[content['engine']['application']])
        else:
            print("Warning: No application found in input. Defaulting to PySCF")
            default_content = deepcopy(defaults['pyscf'])

        self.__dict__.update(default_content)
        self._hash = hashlib.md5(json.dumps(self.__dict__).encode()).hexdigest()

        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                self.__dict__[key].update(content.get(key,{}))
            else:
                self.__dict__[key] = content.get(key, val)

        self._basis = {}
        self._complete = False
        find_projector_type(self)
        fix_basis(self)
        self._complete = True
        self.preprocessor.update(self._basis)

    @property
    def _dict(self):
        return {key: val for key, val in self.__dict__.items() if
            key not in self.private_keys}

    def __getitem__(self, key):
        # if key == 'preprocessor' and self._complete:
        #     # return BasisInstructions(self.preprocessor, self._basis)
        #     return self.preprocessor.update(self._basis)
        # else:
        return self._dict[key]

    def get_hash(self):
        return self._hash

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        # return json.dumps(self._dict, indent=4)
        return json.dumps(self.__dict__ , indent=4)

    def __str__(self):
        return self.__repr__()
