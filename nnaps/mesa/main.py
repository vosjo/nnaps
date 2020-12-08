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


def _compress(args):
    if args.outputdir is None:
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

    infofile = setup.get('infofile', None)
    if args.infofile is not None:
        infofile = args.infofile

    inputdir = setup.get('inputdir', None)
    if args.inputdir is not None:
        inputdir = args.inputdir

    if infofile is None and inputdir is None:
        print("You need to specify an input directory with option -i <path> and/or and info file with option -f <path>")
        exit()

    if infofile is None:
        modelfile = Path(inputdir)

        # check if the path given is of 1 MESA model, or if it contains multiple directories with MESA models
        # check is performed by checking if a modelfile/inlist file exist
        if (modelfile / 'inlist').is_file():
            model_list = [modelfile]
            model_list = pd.DataFrame(data={'path': [p.name for p in model_list]})
            setup['input_path_prefix'] = ''
            setup['input_path_kw'] = 'path'
        else:
            model_list = modelfile.glob('*')
            model_list = pd.DataFrame(data={'path': [p.name for p in model_list]})
            setup['input_path_prefix'] = inputdir
            setup['input_path_kw'] = 'path'
    else:
        model_list = pd.read_csv(infofile)

        # Check if the info file contains the path to the models to compress
        if not 'path' in model_list.columns.values:
            print("The info file needs to contain a column called path containing the path to the MESA model.")
            exit()

        if inputdir is not None:
            setup['input_path_prefix'] = inputdir
        else:
            setup['input_path_prefix'] = './'

    compress_mesa.convert2hdf5(model_list, output_path=args.outputdir, **setup, skip_existing=args.skip, verbose=True)

    print("--> {}".format(args.outputdir))


def _extract(args):
    # if necessary load default setup. Check local folder first, then system defaults, then load from defaults file
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
    output = setup.get('output', None)
    if args.outputfile is not None:
        output = args.outputfile

    if output is None:
        print("You need to specify an output file with option -o <filename>")
        exit()

    # check which input directories to use
    input = setup.get('input', None)
    if args.input is not None:
        input = args.input

    if input is None:
        print("No input directory or csv file specified, nothing to do!")
        exit()

    file_list = get_file_list([input])

    if len(file_list) == 0:
        print("No input files found, nothing to do!")
        exit()
    else:
        print('Found {} files'.format(len(file_list)))

    result = extract_mesa.extract_mesa(file_list, **setup, verbose=True)

    result.to_csv(output, index=False, na_rep='NaN')

    print("--> {}".format(output))

def main():
    description = '''
    Neural Network assistes Population Synthesis code (NNaPS):
    Process stellar evolution models and use machine learning to turn 1D stellar evolution models into binary
    population synthesis models.
    Detailed documentation can be found on https://nnaps.readthedocs.io/ 
    '''

    epilog = '''Author: Joris Vos, send comments and bug reports to: joris.vos@uv.cl'''

    parser = argparse.ArgumentParser(description=description, epilog=epilog)

    subparsers = parser.add_subparsers(dest='action')

    # --compress--
    compress_parser = subparsers.add_parser('compress', help='Compress MESA 1D stellar evolution models to single '
                                                             'hdf5 binary files for easy processing and storage.')

    compress_parser.add_argument('-i, -inputdir', dest='inputdir', default=None,
                                 help='The directory containing all evolution models')
    compress_parser.add_argument('-f, -infofile', dest='infofile', default=None,
                                 help='CSV file containing extra information on the models')
    compress_parser.add_argument('-o, -outputdir', dest='outputdir', default=None,
                              help='The output directory for the compressed models')
    compress_parser.add_argument('-s, -setup', dest='setup', default=None,
                              help='The setup file containing all settings for compression')
    compress_parser.add_argument('--skip', dest='skip', default=False, action='store_true',
                              help='skip models that have already been transformed to h5.')
    compress_parser.set_defaults(func=_compress)

    # --extract--
    extract_parser = subparsers.add_parser('extract',
                                           help='Extract aggregate parameters from compressed evolution models and '
                                                'store them as csv files.')

    extract_parser.add_argument('-i, -input', dest='input', default=None,
                                 help='The directory containing all compressed models in h5 format or a csv file '
                                      'containing the path to all models to extract.')
    extract_parser.add_argument('-o, -outputfile', dest='outputfile', default=None,
                                 help='The output filename for the extracted parameters')
    extract_parser.add_argument('-s, -setup', dest='setup', default=None,
                                 help='The setup file containing all settings for extraction')
    extract_parser.set_defaults(func=_extract)

    args = parser.parse_args()
    args.func(args)


if __name__=="__main__":
    main()
