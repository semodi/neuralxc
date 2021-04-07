"""
cp2k.py
Extends the ASE CP2K calculator, by letting it read in custom input files.
In particular it allows to add NeuralXC type functionals in the LIBXC section.
Some code in this file was adapted from the original ASE implementation.
"""
import re

from ase.calculators.cp2k import CP2K


class CustomCP2K(CP2K):
    """ extends ASE calculator's ability to include NXC functionals
    """
    def __init__(self, **kwargs):
        nxc_default = {'path': "", 'add_to': ""}
        self.nxc = kwargs.pop("nxc", nxc_default)
        self.nxc_addto = self.nxc["add_to"]
        self.nxc = self.nxc["path"]
        self.input_path = kwargs.pop("input_path", "")
        inp = ''
        if not 'command' in kwargs:
            kwargs['command'] = 'env OMP_NUM_THREADS=1 cp2k_shell.sdbg'
        if self.input_path:
            with open(self.input_path, 'r') as inp_file:
                inp = inp_file.read()
        kwargs['inp'] = inp
        super().__init__(**kwargs)

    def _generate_input(self, *args, **kwargs):
        input = super()._generate_input(*args, **kwargs)

        if self.nxc:
            nxc = self.nxc
            nxc_addto = self.nxc_addto

            pattern = re.compile("LIBXC.*?{}.*?END LIBXC".format(nxc_addto), re.MULTILINE | re.S)

            pattern0 = pattern.findall(input)[0]

            pattern1 = pattern0.replace('{}\n'.format(nxc_addto), '{}\n \t\tNXC {}\n'.format(nxc_addto, nxc))

            input = input.replace(pattern0, pattern1)
        return input
