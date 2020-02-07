import os
import h5py

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

    @param filename: the name of the hdf5 file to read
    @type filename: str
    @return: dictionary with read filestructure
    @rtype: dict
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
