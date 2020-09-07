import os
import h5py

import numpy as np
from numpy.lib.recfunctions import append_fields

from scipy.interpolate import interp1d


def write2hdf5(data, filename, update=False, attr_types=[]):
    """
    Write the content of a dictionary to a hdf5 file. The dictionary can contain other
    nested dictionaries, this file stucture will be maintained in the saved hdf5 file.

    Pay attention to the fact that the data type of lists might change when writing to
    hdf5. Lists are stored as numpy arrays, thus all items in a list are converted to
    the same type: ['bla', 1, 24.5] will become ['bla', '1', '24.5']. Upt till now there
    is nothing in place to check this, or correct it when reading a hdf5 file.

    @param data: the dictionary to write to file
    @type data: dict
    @param filename: the name of the hdf5 file to write to
    @type filename: str
    @param update: True if you want to update an existing file, False to overwrite
    @type update: bool
    @param attr_types: the data types that you want to save as an attribute instead of
                      a dataset. (standard everything is saved as dataset.)
    @type attr_types: List of types
    """

    if not update and os.path.isfile(filename):
        os.remove(filename)

    def save_rec(data, hdf):
        """ recursively save a dictionary """
        for key in data.keys():
            try:

                if type(data[key]) == dict:
                    # if part is dictionary: add 1 level and save dictionary in new level
                    if not key in hdf:
                        hdf.create_group(key)
                    save_rec(data[key], hdf[key])

                elif type(data[key]) in attr_types:
                    # save data as attribute
                    hdf.attrs[key] = data[key]

                else:
                    # other data is stored as datasets
                    if key in hdf:
                        del hdf[key]
                    hdf.create_dataset(key, data=data[key])

            except Exception as e:
                print( 'Error while trying to write: {}, type: {}'.format(key, type(key)) )
                raise(e)

    hdf = h5py.File(filename)
    save_rec(data, hdf)
    hdf.close()


def read_hdf5(filename):
    """
    Read the filestructure of a hdf5 file to a dictionary.

    Iteratively read a hdf5 file and return the content as a dictionary. Can be used to read the h5 files created
    with the mesa -2h5 method. If you want to read the compressed mesa models and receive the data in a directly usable
    format, you can use the :func:`~nnaps.mesa.fileio.read_compressed_track` function

    :param filename: the name of the hdf5 file to read
    :type filename: str
    :return: dictionary with the content of the hdf5 file
    :rtype: dict
    """

    if not os.path.isfile(filename):
        print("File does not exist")
        raise IOError

    def read_rec(hdf):
        """ recursively read the hdf5 file """
        res = {}
        for name, grp in hdf.items():
            # -- read the subgroups and datasets
            if hasattr(grp, 'items'):
                # in case of a group, read the group into a new dictionary key
                res[name] = read_rec(grp)
            else:
                # in case of dataset, read the value
                res[name] = grp.value

        # -- read all the attributes
        for name, atr in hdf.attrs.items():
            res[name] = atr

        return res

    hdf = h5py.File(filename, 'r')
    result = read_rec(hdf)
    hdf.close()

    return result


def read_compressed_track(filename, return_profiles=False):
    """
    Function to read a compressed hdf5 model. It will automatically combine the evolution history of the stellar parts
    and the binary part in one numpy rec array, while correcting for potentially different model numbers in the
    different history files. It will also return any extra information included by the mesa-2h5 command as a dictionary.
    If :option:`return_profiles` is set to True, it will also return a dictionary containing all profiles together
    with a dictionary mapping the different profile names to the model number at which they were created.

    **Combining history**:
        Currently the binary history file is taken as the base to determine which model numbers will be part of the
        final evolution history. The stellar history data of the primary and secondary are then interpolated to match
        the model numbers of the binary history.

        To avoid naming conflicts, the history parameters of the secondary get a '_2' added to their name.

        This function also adds several extra parameters to the history file if they can be inferred from other
        parameters. This is because later functions that derive stability ect might require these. Derived parameters
        are: effective_T, effective_T_2, rl_overflow_1, mass_ratio, separation_au, log10_J_div_Jdot_div_P and
        log10_M_div_Mdot_div_P. It also adds a column called CE_phase which defaults to 0, as this is required for
        the stability and CE phase determination later on.

    **Profiles**:
        If profiles have to be returned (:option:`return_profiles = True`), they are returned in a dictionary. This
        dictionary contains all profiles by name, and a legend called 'profile_legend'. This legend contains a mapping
        between all included profile names and the model number at which time they were taken.

        .. code-block:: python

            profiles = {'profile_1': np.recarray(),
                        'profile_2': np.recarray(),
                        'profile_legend' = {'profile_1': 150, 'profile_2': 329},
                       }


    :param filename: The path to the hdf5 compressed file to read
    :type filename: str
    :param return_profiles: If True, return a dictionary containing the profiles.
    :type return_profiles: bool
    :return: history, extra_info (, profiles): A numpy rec array containing the combined history, a dictionary with
             any extra info, and optionally a dictionary containing all profiles.
    :rtype: rec_array, dict (, dict)
    """

    data_ = read_hdf5(filename)

    history = data_.pop('history', None)
    d1 = history.pop('star1', None)
    d2 = history.pop('star2', None)
    db = history.pop('binary', None)


    extra_info = data_.pop('extra_info', None)

    # set model number for primary to start at 1 and limits to correct last model number
    d1['model_number'] = d1['model_number'] - d1['model_number'][0] + 1
    s = np.where(db['model_number'] <= d1['model_number'][-1])
    db = db[s]

    # PRIMARY
    # now interpolate primary data to match model numbers for binary history
    dtypes = d1.dtype
    y = d1.view(np.float64).reshape(-1, len(dtypes))
    f = interp1d(d1['model_number'], y, axis=0, bounds_error=False, fill_value=0.0)
    d1 = f(db['model_number'])

    # reconvert from array to recarray
    d1 = [tuple(d) for d in d1]
    d1 = np.array(d1, dtype=dtypes)

    # remove model_number as column from d1 and merge into 1 recarray
    columns1 = list(d1.dtype.names)
    columns1.remove('model_number')
    column_names1 = [c for c in columns1]

    # SECONDARY
    if d2 is not None:
        # now interpolate secondary data to match model numbers for binary history
        dtypes = d2.dtype
        y = d2.view(np.float64).reshape(-1, len(dtypes))
        f = interp1d(d2['model_number'], y, axis=0, bounds_error=False, fill_value=0.0)
        d2 = f(db['model_number'])

        # reconvert from array to recarray
        d2 = [tuple(d) for d in d2]
        d2 = np.array(d2, dtype=dtypes)

        # remove model_number as column from d1 and merge into 1 recarray
        columns2 = list(d2.dtype.names)
        columns2.remove('model_number')
        column_names2 = [c+'_2' for c in columns2]

    # create a new record array from the data (much faster than appending to an existing array)
    columnsdb = list(db.dtype.names)

    if d2 is not None:
        all_data = [db[c] for c in columnsdb] + [d1[c] for c in columns1] + [d2[c] for c in columns2]
        all_columns = columnsdb + column_names1 + column_names2
    else:
        all_data = [db[c] for c in columnsdb] + [d1[c] for c in columns1]
        all_columns = columnsdb + column_names1

    data = np.core.records.fromarrays(all_data, names=all_columns)

    fields = []
    fields_data = []

    if not 'effective_T' in all_columns and 'log_Teff' in all_columns:
        fields.append('effective_T')
        fields_data.append(10**data['log_Teff'])

    if not 'effective_T_2' in all_columns and 'log_Teff_2' in all_columns:
        fields.append('effective_T_2')
        fields_data.append(10**data['log_Teff_2'])

    if not 'rl_overflow_1' in all_columns and 'star_1_radius' in all_columns and 'rl_1' in all_columns:
        fields.append('rl_overflow_1')
        fields_data.append(data['star_1_radius'] / data['rl_1'])

    if not 'mass_ratio' in all_columns and 'star_1_mass' in all_columns and 'star_2_mass' in all_columns:
        fields.append('mass_ratio')
        fields_data.append(data['star_1_mass'] / data['star_2_mass'])

    if not 'separation_au' in all_columns and 'binary_separation' in all_columns:
        fields.append('separation_au')
        fields_data.append(data['binary_separation'] * 0.004649183820234682)

    if not 'CE_phase' in all_columns and db is not None:
        # only add when the model is binary
        fields.append('CE_phase')
        fields_data.append(np.zeros_like(data['model_number']))

    if 'J_orb' in all_columns and 'Jdot' in all_columns and 'period_days' in all_columns:
        J_Jdot_P = (data['J_orb'] / np.abs(data['Jdot'])) / (data['period_days'] * 24.0 *60.0 *60.0)
        J_Jdot_P = np.where((J_Jdot_P == 0 ), 99, np.log10(J_Jdot_P))
        fields.append('log10_J_div_Jdot_div_P')
        fields_data.append(J_Jdot_P)

    if 'star_1_mass' in all_columns and 'lg_mstar_dot_1' in all_columns and 'period_days' in all_columns:
        M_Mdot_P = (data['star_1_mass'] / 10 ** data['lg_mstar_dot_1']) / (data['period_days'] / 360)
        M_Mdot_P = np.where((M_Mdot_P == 0), 99, np.log10(M_Mdot_P))
        fields.append('log10_M_div_Mdot_div_P')
        fields_data.append(M_Mdot_P)

    data = append_fields(data, fields, fields_data, usemask=False)

    if return_profiles:
        if 'profiles' not in data_:
            profiles = None
        else:
            profiles = data_.get('profiles')
            profiles['legend'] = data_.get('profile_legend')

        return data, extra_info, profiles

    return data, extra_info
