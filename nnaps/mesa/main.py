import os
import glob
import argparse
import pandas as pd

from pathlib import Path

from nnaps.mesa import compress_mesa, extract_mesa, defaults


def get_file_list(input_list):
    """
    Function that returns a list of all models that should be extracted. Input can be a list of directories, a csv file
    containing the path to all models, or a list of csv files containing the path to the models to include.

    If the input is given as csv files, each csv file should at least contain 1 column named 'path' containing the
    path to the MESA model to process. The csv file can also contain columns with extra parameters that can be relevant
    during the model extraction.

    :param input_list: list of input directories or csv files
    :return: pandas dataframe containing the path to all models to process
    """

    # differentiate between csv files and directories
    if os.path.isdir(input_list[0]):
        files = []
        for input_dir in input_list:
            files_ = glob.glob(str(Path(input_dir, '*')))
            files += files_

        file_list = pd.DataFrame(data=files, columns=['path'])

    else:
        # all inputs are csv files
        files = []
        for input_file in input_list:
            d = pd.read_csv(input_file)
            files.append(d)

        file_list = pd.concat(files)

    return file_list


def main():
    parser = argparse.ArgumentParser(description='NNaPS-mesa: Process MESA models')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-2h5', dest='toh5', nargs='+', default=None,
                        help='Convert MESA models history files to h5 format')
    group.add_argument('-extract', dest='extract', default=None, nargs='*',
                        help='Extract parameters from history files stored as h5')
    parser.add_argument('-setup', dest='setup', default=None,
                        help='The setup file containing necessary info for the -2h5 and -extract option')
    parser.add_argument('-o', dest='output', default=None,
                        help='The output file or directory for the -2h5 and -extract functions')
    parser.add_argument('--skip', dest='skip', default=False, action='store_true',
                        help='For 2h5: skip models that have already been transformed to h5.')
    args = parser.parse_args()

    if args.toh5 is not None:
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

        modelfile = Path(args.toh5[0])

        if modelfile.is_dir():
            # check if the path given is of 1 MESA model, or if it contains multiple directories with MESA models
            # check is performed by checking if a modelfile/inlist file exist

            if (modelfile / 'inlist' ).is_file():
                model_list = [modelfile]
                model_list = pd.DataFrame(data={'path': [p.name for p in model_list]})
                setup['input_path_prefix'] = ''
                setup['input_path_kw'] = 'path'
            else:
                model_list = modelfile.glob('*')
                model_list = pd.DataFrame(data={'path': [p.name for p in model_list]})
                setup['input_path_prefix'] = args.toh5[0]
                setup['input_path_kw'] = 'path'
        else:
            model_list = pd.read_csv(args.toh5[0])
            if len(args.toh5) > 1:
                setup['input_path_prefix'] = args.toh5[1]

        compress_mesa.convert2hdf5(model_list, output_path=args.output, **setup, skip_existing=args.skip, verbose=True)

        print("--> {}".format(args.output))

    elif args.extract is not None:

        if args.setup is None:

            if os.path.isfile('default_extract.yaml'):
                setup = defaults.read_defaults('default_extract.yaml')
            elif os.path.isfile('~/.nnaps/default_extract.yaml'):
                setup = defaults.read_defaults('~/.nnaps/default_extract.yaml')
            else:
                setup = defaults.default_extract

        else:
            setup = defaults.read_defaults(args.setup)

        # check which output filename to use
        if 'output' not in setup and args.output is None:
            print("You need to specify an output file with option -o <filename>")
            exit()
        else:
            if args.output is not None:
                output = args.output
            else:
                output = setup['output']

        # check which input directories to use
        if 'input' in setup and len(args.extract) == 0:
            file_list = get_file_list(setup['input'])
        else:
            file_list = get_file_list(args.extract)

        if len(file_list) == 0:
            print("No input files found!")
            exit()
        else:
            print('Found {} files'.format(len(file_list)))

        result = extract_mesa.extract_mesa(file_list, **setup, verbose=True)

        result.to_csv(output, index=False, na_rep='NaN')

        print("--> {}".format(output))

    else:
        print("Nothing to do!\nUse as:\n"
              ">>> nnaps-mesa -2h5 <modelfile.csv> <input_path> -o <output_path>\n"
              ">>> nnaps-mesa -extract <input_path> -o <output_file>\n"
              "For help run\n>>> nnaps-mesa -h")


if __name__=="__main__":
    main()
