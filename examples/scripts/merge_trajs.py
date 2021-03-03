import sys

from ase.io import read, write

all_sys = []

for path in sys.argv[1:]:
    all_sys += read(path, ':')

write('merged.traj', all_sys)
