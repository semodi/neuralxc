import sys

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.units import kcal, mol

kcalpmol = kcal / mol


def get_structures_energies(path, unit=1):
    """
    Loads an xyz file that contains energies in the comment line 
    and returns an ase Atoms list
    """
    content = np.genfromtxt(path, delimiter='###')
    energies = content[1::int(content[0] + 2)] * unit

    atoms = read(path, ':')

    for a, e in zip(atoms, energies):
        a.calc = SinglePointCalculator(a)
        a.calc.results['energy'] = e

    return atoms


if __name__ == '__main__':

    if len(sys.argv) == 5:
        unit = float(len(sys.argv[4]))
    else:
        unit = kcalpmol

    atoms = get_structures_energies(sys.argv[1], unit=kcalpmol)
    if not sys.argv[2] == 'all':
        subset = np.genfromtxt(sys.argv[2]).astype(int)
    else:
        subset = np.arange(len(atoms))
    atoms_subset = [atoms[s] for s in subset]
    write(sys.argv[3], atoms_subset)
