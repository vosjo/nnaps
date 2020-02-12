import os
import yaml
import glob
import argparse
import pandas as pd

from pathlib import Path

from . import read_mesa, extract_mesa, defaults

def main():
    parser = argparse.ArgumentParser(description='NNaPS-mesa: Process MESA models')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-2h5', dest='modelfile', nargs='+', default=None,
                        help='Convert MESA models history files to h5 format')
    group.add_argument('-extract', dest='extract', default=None,
                        help='Extract parameters from history files stored as h5')
    parser.add_argument('-setup', dest='setup', nargs=1, default=None,
                        help='The setup file containing necessary info for the -2h5 and -extract option')
    parser.add_argument('-o', dest='output', default=None,
                        help='The output file or directory for the -2h5 and -extract functions')
    parser.add_argument('--skip', dest='skip', default=False, action='store_true',
                        help='For 2h5: skip models that have already been transformed to h5.')
    args = parser.parse_args()

    if args.modelfile is not None:
        # run the convert2hdf5 function
        model_list = pd.read_csv(args.modelfile[0])

        # if necessary load default setup. Check local folder first, then system defaults, then load from defaults file
        if args.setup is None:

            if os.path.isfile('default_2h5.yaml'):
                setup = defaults.read_defaults('default_2h5.yaml')
            elif os.path.isfile('~/.nnaps/default_2h5.yaml'):
                setup = defaults.read_defaults('~/.nnaps/default_2h5.yaml')
            else:
                setup = defaults.defaults_2h5
        else:
            setup = defaults.read_defaults(args.setup)

        if len(args.modelfile) > 1:
            setup['input_path_prefix'] = args.modelfile[1]

        read_mesa.convert2hdf5(model_list, output_path=args.output, **setup, skip_existing=args.skip, verbose=True)

    elif args.extract is not None:

        files = glob.glob(str(Path(args.extract, '*')))
        file_list = pd.DataFrame(data=files, columns=['path'])

        if args.setup is None:

            if os.path.isfile('default_extract.yaml'):
                setup = yaml.safe_load('default_extract.yaml')
            elif os.path.isfile('~/.nnaps/default_extract.yaml'):
                setup = yaml.safe_load('~/.nnaps/default_extract.yaml')
            else:
                setup = defaults.default_extract
        else:
            setup = yaml.safe_load(args.setup)

        result = extract_mesa.extract_mesa(file_list, **setup, verbose=True)

        result.to_csv(args.output)

    else:
        print("Nothing to do!\nUse as:\n"
              ">>> nnaps-mesa -2h5 <modelfile.csv> <input_path> -o <output_path>\n"
              ">>> nnaps-mesa -extract <input_path> -o <output_file>")