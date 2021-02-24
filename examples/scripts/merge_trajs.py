from ase.io import read, write
import sys

all_sys = []

for path in sys.argv[1:]:
    all_sys += read(path, ':')

write('merged.traj', all_sys)
