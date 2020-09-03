import os
import glob
import argparse
import pandas as pd

from pathlib import Path

from nnaps.mesa import read_mesa, extract_mesa, defaults

#Check Gooey for a very simple gui for nnaps-mesa
from gooey import Gooey, GooeyParser
# https://github.com/chriskiehl/Gooey

def get_file_list(input_dirs):

    files = []

    for input_dir in input_dirs:
        files_ = glob.glob(str(Path(input_dir, '*')))
        files += files_

    file_list = pd.DataFrame(data=files, columns=['path'])

    return file_list


def compress(inputdir, setupfile, inputfile, outputdir, skip=False):
    # run the convert2hdf5 function

    print ("Compress", inputdir, setupfile, inputfile, outputdir, skip)

    # if output is None:
    #     print("You need to specify an output path with option -o <path>")
    #     exit()
    #
    # # if necessary load default setup. Check local folder first, then system defaults, then load from defaults file
    # if setup is None:
    #     if os.path.isfile('default_2h5.yaml'):
    #         setup = defaults.read_defaults('default_2h5.yaml')
    #     elif os.path.isfile('~/.nnaps/default_2h5.yaml'):
    #         setup = defaults.read_defaults('~/.nnaps/default_2h5.yaml')
    #     else:
    #         setup = defaults.defaults_2h5
    # else:
    #     setup = defaults.read_defaults(setup)
    #
    # modelfile = Path(input[0])
    #
    # if modelfile.is_dir():
    #     # check if the path given is of 1 MESA model, or if it contains multiple directories with MESA models
    #     # check is performed by checking if a modelfile/inlist file exist
    #
    #     if (modelfile / 'inlist').is_file():
    #         model_list = [modelfile]
    #         model_list = pd.DataFrame(data={'path': [p.name for p in model_list]})
    #         setup['input_path_prefix'] = ''
    #         setup['input_path_kw'] = 'path'
    #     else:
    #         model_list = modelfile.glob('*')
    #         model_list = pd.DataFrame(data={'path': [p.name for p in model_list]})
    #         setup['input_path_prefix'] = args.toh5[0]
    #         setup['input_path_kw'] = 'path'
    # else:
    #     model_list = pd.read_csv(input)
    #     if len(input) > 1:
    #         setup['input_path_prefix'] = input[1]
    #
    # read_mesa.convert2hdf5(model_list, output_path=outputdir, **setup, skip_existing=skip, verbose=True)
    #
    # print("--> {}".format(outputdir))

def extract(inputdir, setupfile, outputfile):

    print ("Extract", inputdir, setupfile, outputfile)

    # if setupfile is None:
    #
    #     if os.path.isfile('default_extract.yaml'):
    #         setup = defaults.read_defaults('default_extract.yaml')
    #     elif os.path.isfile('~/.nnaps/default_extract.yaml'):
    #         setup = defaults.read_defaults('~/.nnaps/default_extract.yaml')
    #     else:
    #         setup = defaults.default_extract
    #
    # else:
    #     setup = defaults.read_defaults(setupfile)
    #
    # # check which output filename to use
    # if 'output' not in setup and outputfile is None:
    #     print("You need to specify an output file with option -o <filename>")
    #     exit()
    # else:
    #     if outputfile is not None:
    #         output = outputfile
    #     else:
    #         output = setup['output']
    #
    # # check which input directories to use
    # if 'input' in setup and len(inputdir) == 0:
    #     file_list = get_file_list(setup['input'])
    # else:
    #     file_list = get_file_list(inputdir)
    #
    # if len(file_list) == 0:
    #     print("No input files found!")
    #     exit()
    # else:
    #     print('Found {} files'.format(len(file_list)))
    #
    # result = extract_mesa.extract_mesa(file_list, **setup, verbose=True)
    #
    # result.to_csv(output, index=False, na_rep='NaN')
    #
    # print("--> {}".format(output))

@Gooey(advanced=True,
       required_cols=1,  # number of columns in the "Required" section
       optional_cols=1,  # number of columns in the "Optional" section
       )
def main():
    parser = GooeyParser(description='NNaPS-mesa: Process MESA models')

    subparsers = parser.add_subparsers(dest='action')

    #--Compress--

    compress_parser = subparsers.add_parser('Compress', help='Compress a grid of MESA models')

    input_group = compress_parser.add_argument_group("Input")

    input_group.add_argument('inputdir', default=None,
                        help='Directory containing the MESA models to process',
                        widget='DirChooser')
    input_group.add_argument('-inputfile', default=None,
                        help='CSV file containing extra info for models to process',
                        widget='FileChooser')
    input_group.add_argument('setupfile', default=None,
                        help='The setup file containing necessary settings',
                        widget='FileChooser')
    input_group.add_argument('outputdir', default=None,
                        help='The output directory to store the compressed models',
                        widget='DirChooser')

    option_group = compress_parser.add_argument_group("Options")

    option_group.add_argument('--skip', dest='skip', default=False, action='store_true',
                                help='skip models that have already been transformed to h5.')

    #--Extract--

    extract_parser = subparsers.add_parser("Extract",
                                        help="Extract aggregate parameters from a grid of compressed MESA models")

    input_group_2 = extract_parser.add_argument_group("Input")

    input_group_2.add_argument('-inputdir', default=None,
                        help='Directory containing the compressed MESA models to process',
                        widget='DirChooser')
    input_group_2.add_argument('setupfile', default=None,
                        help='The setup file containing necessary settings',
                        widget='FileChooser')
    input_group_2.add_argument('-outputfile', default=None,
                        help='The output csv file to store the extracted parameters')

    args = parser.parse_args()

    if args.action == 'Compress':

        compress(args.inputdir, args.setupfile, args.inputfile, args.outputdir, skip=args.skip)

    else:

        extract(args.inputdir, args.setupfile, args.outputfile)


def main_old():
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

        read_mesa.convert2hdf5(model_list, output_path=args.output, **setup, skip_existing=args.skip, verbose=True)

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
