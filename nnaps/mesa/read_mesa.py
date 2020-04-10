import os
import glob
import warnings


from pathlib import Path
import numpy as np

# repack_fields is necessary since np 1.16 as selecting columns from a recarray returns an array with padding
# that is difficult to work with afterwards.
from numpy.lib import recfunctions as rf

from . import fileio


def read_mesa_output(filename=None, only_first=False):
    """
    Read star.log and .data files from MESA.

    This returns a record array with the global and local parameters (the latter
    can also be a summary of the evolutionary track instead of a profile if
    you've given a 'star.log' file.

    The stellar profiles are given from surface to center.

    Function writen by Pieter DeGroote

    @param filename: name of the log file
    @type filename: str
    @param only_first: read only the first model (or global parameters)
    @type only_first: bool
    @return: list of models in the data file (typically global parameters, local parameters)
    @rtype: list of rec arrays
    """
    models = []
    new_model = False
    header = None
    # -- open the file and read the data
    with open(filename, 'r') as ff:
        # -- skip first 5 lines when difference file
        if os.path.splitext(filename)[1] == '.diff':
            for i in range(5):
                line = ff.readline()
            models.append([])
            new_model = True
        while 1:
            line = ff.readline()
            if not line:
                break  # break at end-of-file
            line = line.strip().split()
            if not line:
                continue
            # -- begin a new model
            if all([iline == str(irange) for iline, irange in zip(line, range(1, len(line) + 1))]):
                # -- wrap up previous model
                if len(models):
                    model = np.array(models[-1], float).T
                    models[-1] = np.rec.fromarrays(model, names=header)
                    if only_first: break
                models.append([])
                new_model = True
                continue
            # -- next line is the header of the data, remember it
            if new_model:
                header = line
                new_model = False
                continue
            models[-1].append(line)
    if len(models) > 1:

        try:
            model = np.array(models[-1], float).T
        except:
            indices = []
            for i, l in enumerate(models[-1]):
                if len(l) != len(models[-1][0]):
                    indices.append(i)

            for i in reversed(indices):
                del models[-1][i]
            print("Found and fixed errors on following lines: ", indices)
            model = np.array(models[-1], float).T

        models[-1] = np.rec.fromarrays(model, names=header)

    return models


def get_end_log_file(logfile):
    if os.path.isfile(logfile):
        # case for models ran locally
        ifile = open(logfile)
        lines = ifile.readlines()
        ifile.close()

        return lines[-30:-1]
    else:
        return []


def convert2hdf5(modellist, star_columns=None, binary_columns=None, profile_columns=None,
                 add_stopping_condition=True, skip_existing=True,
                 star1_history_file='LOGS/history1.data', star2_history_file='LOGS/history2.data',
                 binary_history_file='LOGS/binary_history.data', log_file='log.txt',
                 profile_files=None, profiles_path='', profile_pattern='*.profile',
                 input_path_kw='path', input_path_prefix='', output_path=None, verbose=False):

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    for i, model in modellist.iterrows():

        print(input_path_prefix, model[input_path_kw])

        if not os.path.isdir(Path(input_path_prefix, model[input_path_kw])):
            continue

        if skip_existing and os.path.isfile(Path(output_path, model[input_path_kw]).with_suffix('.h5')):
            if verbose:
                print(i, model[input_path_kw], ': exists, skipping')
            continue

        if verbose:
            print(i, model[input_path_kw], ': processing')

        # store all columns of the input file in the hdf5 file
        data = {}
        extra_info = {}
        for col in model.index:
            extra_info[col] = model[col]

        # obtain the termination code and store if requested
        termination_code = 'uk'
        if add_stopping_condition:
            lines = get_end_log_file(Path(input_path_prefix, model[input_path_kw], log_file))
            for line in lines:
                if 'termination code' in line:
                    termination_code = line.split()[-1]

        extra_info['termination_code'] = termination_code

        data['extra_info'] = extra_info

        # check if all history files that are requested are available and can be read. If there is an error,
        # skip to the next model
        history = {}
        if star1_history_file is not None:
            try:
                d1 = read_mesa_output(Path(input_path_prefix, model[input_path_kw], star1_history_file))[1]
                if star_columns is not None:
                    d1 = rf.repack_fields(d1[star_columns])
                history['star1'] = d1
            except Exception as e:
                if verbose:
                    print("Error in reading star1: ", e)
                continue

        if star2_history_file is not None:
            try:
                d2 = read_mesa_output(Path(input_path_prefix, model[input_path_kw], star2_history_file))[1]
                if star_columns is not None:
                    d2 = rf.repack_fields(d2[star_columns])
                history['star2'] = d2
            except Exception as e:
                if verbose:
                    print("Error in reading star2: ", e)
                continue

        if binary_history_file is not None:
            try:
                d3 = read_mesa_output(Path(input_path_prefix, model[input_path_kw], binary_history_file))[1]
                if star_columns is not None:
                    d3 = rf.repack_fields(d3[binary_columns])
                history['binary'] = d3
            except Exception as e:
                if verbose:
                    print("Error in reading binary: ", e)
                continue

        data['history'] = history

        # check if profiles exists and store them is requested. Also make a profile lookup table (legend)
        profiles = {}
        profile_legend = []
        profile_name_length = 0 # store longest profile name to create recarray of profile_legend
        if profile_files is not None:
            if profile_files == 'all':
                profile_paths = Path(input_path_prefix, model[input_path_kw], profiles_path).glob(profile_pattern)
            else:
                profile_paths = [Path(input_path_prefix, model[input_path_kw], profiles_path, p) for p in profile_files]

            for filepath in profile_paths:
                if not filepath.is_file():
                    continue

                profile_name = filepath.stem
                header, profile_data = read_mesa_output(filename=filepath, only_first=False)

                if profile_columns is not None:
                    profile_data = rf.repack_fields(profile_data[profile_columns])
                profiles[profile_name] = profile_data

                if len(profile_name) > profile_name_length:
                    profile_name_length = len(profile_name)
                profile_legend.append((header['model_number'], profile_name))

        if len(profiles.keys()) >= 1:
            data['profiles'] = profiles
            profile_legend = np.array(profile_legend, dtype=[('model_number', 'f8'),
                                                             ('profile_name', 'a'+str(profile_name_length))])
            data['profile_legend'] = profile_legend

        # rather annoying way to assure that Path doesn't cut of part of the folder name when adding the .h5 suffix
        # if not this will happen: M1.080_M0.502_P192.67_Z0.01129 -> M1.080_M0.502_P192.67_Z0.h5
        output_file = Path(output_path, model[input_path_kw])
        output_file = output_file.with_suffix(output_file.suffix + '.h5')
        fileio.write2hdf5(data, output_file, update=False)
