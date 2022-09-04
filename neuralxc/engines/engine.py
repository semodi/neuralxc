"""
engines.py
Engines that act as adapters between NeuralXC and electronic structure codes (or their
ASE calculator class if it exists). Used for dataset creation and passed to driver.
"""
from abc import abstractmethod

from ..base import ABCRegistry

try:
    from pyscf.scf.chkfile import load_scf

    from neuralxc.pyscf.pyscf import compute_KS
except ModuleNotFoundError:
    compute_KS = None
import os

from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Hartree

from neuralxc.engines.cp2k import CustomCP2K
from neuralxc.engines.siesta import CustomSiesta


class EngineRegistry(ABCRegistry):
    REGISTRY = {}


class BaseEngine(metaclass=EngineRegistry):

    _registry_name = 'base'

    @abstractmethod
    def compute(self, atoms):
        pass


def Engine(app, **kwargs):

    if app == 'pyscf_rad':
        app = 'pyscf'
    registry = BaseEngine.get_registry()
    if app not in registry:
        raise Exception(f'Engine: {app} not registered')

    return registry[app](**kwargs)


class PySCFEngine(BaseEngine):

    _registry_name = 'pyscf'

    def __init__(self, **kwargs):
        self.xc = kwargs.pop('xc', 'PBE')
        self.basis = kwargs.pop('basis', 'ccpvdz')
        self.nxc = kwargs.pop('nxc', '')
        self.skip_calculated = kwargs.pop('skip_calculated', True)
        self.engine_kwargs = kwargs

    def compute(self, atoms):
        if 'pyscf.chkpt' in os.listdir('.') and self.skip_calculated:
            print('Re-using results')
            mol, results = load_scf('pyscf.chkpt')
            e = results['e_tot']
        else:
            mf, mol = compute_KS(atoms, basis=self.basis, xc=self.xc, nxc=self.nxc, **self.engine_kwargs)
            e = mf.energy_tot()

        atoms.calc = SinglePointCalculator(atoms)
        atoms.calc.results = {'energy': e * Hartree}
        return atoms


class PySCFEngine(PySCFEngine):

    _registry_name = 'pyscf_rad'


class ASECalcEngine(BaseEngine):

    _registry_name = 'ase'

    def compute(self, atoms):
        atoms.calc = self.calc
        atoms.get_potential_energy()
        return atoms


class CP2KEngine(ASECalcEngine):

    _registry_name = 'cp2k'

    def __init__(self, **kwargs):
        self.calc = CustomCP2K(**kwargs)


class SiestaEngine(ASECalcEngine):

    _registry_name = 'siesta'

    def __init__(self, **kwargs):
        fdf_path = kwargs.pop('fdf_path', None)

        # Defaults
        kwargs['label'] = kwargs.get('label', 'siesta')
        kwargs['xc'] = kwargs.get('xc', 'PBE')
        kwargs['basis_set'] = kwargs.pop('basis', 'DZP')
        kwargs['fdf_arguments'] = kwargs.get('fdf_arguments', {'MaxSCFIterations': 200})
        kwargs['pseudo_qualifier'] = kwargs.get('pseudo_qualifier', '')
        exec_prepend = kwargs.pop('exec_prepend', '')

        # Environment variables for ase
        os.environ['SIESTA_PP_PATH'] = kwargs.pop('pseudoloc', '.')
        if 'SIESTA_COMMAND' not in os.environ:
            os.environ['SIESTA_COMMAND'] = f'{exec_prepend} siesta < ./%s > ./%s'

        self.calc = CustomSiesta(fdf_path, **kwargs)
