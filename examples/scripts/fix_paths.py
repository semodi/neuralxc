import json
import os

basis = json.load(open('basis_sgdml_benzene.json', 'r'))
basis['engine_kwargs']['pseudoloc'] = os.path.abspath(basis['engine_kwargs']['pseudoloc'])
basis['engine_kwargs']['fdf_path'] = os.path.abspath(basis['engine_kwargs']['fdf_path'])

json.dump(basis, open('basis_sgdml_benzene.json', 'w'), indent=4)
