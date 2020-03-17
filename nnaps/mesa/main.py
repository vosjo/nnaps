import os
import glob
import argparse
import pandas as pd

from pathlib import Path

from nnaps.mesa import read_mesa, extract_mesa, defaults


def main():
    parser = argparse.ArgumentParser(description='NNaPS-mesa: Process MESA models')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-2h5', dest='modelfile', nargs='+', default=None,
                        help='Convert MESA models history files to h5 format')
    group.add_argument('-extract', dest='extract', default=None,
                        help='Extract parameters from history files stored as h5')
    parser.add_argument('-setup', dest='setup', default=None,
                        help='The setup file containing necessary info for the -2h5 and -extract option')
    parser.add_argument('-o', dest='output', default=None,
                        help='The output file or directory for the -2h5 and -extract functions')
    parser.add_argument('--skip', dest='skip', default=False, action='store_true',
                        help='For 2h5: skip models that have already been transformed to h5.')
    args = parser.parse_args()

    if args.modelfile is not None:
        # run the convert2hdf5 function

        if args.output is None:
            print("You need to specify an output path with option -o <path>")
            exit()

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

        modelfile = Path(args.modelfile[0])

        if modelfile.is_dir():
            model_list = modelfile.glob('*')
            model_list = pd.DataFrame(data={'path': [p.name for p in model_list]})
            setup['input_path_prefix'] = args.modelfile[0]
            setup['input_path_kw'] = 'path'
        else:
            model_list = pd.read_csv(args.modelfile[0])
            if len(args.modelfile) > 1:
                setup['input_path_prefix'] = args.modelfile[1]

        read_mesa.convert2hdf5(model_list, output_path=args.output, **setup, skip_existing=args.skip, verbose=True)

    elif args.extract is not None:

        if args.output is None:
            print("You need to specify an output file with option -o <filename>")
            exit()

        files = glob.glob(str(Path(args.extract, '*')))
        file_list = pd.DataFrame(data=files, columns=['path'])

        if args.setup is None:

            if os.path.isfile('default_extract.yaml'):
                setup = defaults.read_defaults('default_extract.yaml')
            elif os.path.isfile('~/.nnaps/default_extract.yaml'):
                setup = defaults.read_defaults('~/.nnaps/default_extract.yaml')
            else:
                setup = defaults.default_extract

        else:
            setup = defaults.read_defaults(args.setup)

        result = extract_mesa.extract_mesa(file_list, **setup, verbose=True)

        result.to_csv(args.output, index=False, na_rep='NaN')

    else:
        print("Nothing to do!\nUse as:\n"
              ">>> nnaps-mesa -2h5 <modelfile.csv> <input_path> -o <output_path>\n"
              ">>> nnaps-mesa -extract <input_path> -o <output_file>\n"
              "For help run\n>>> nnaps-mesa -h")


if __name__=="__main__":
    main()
