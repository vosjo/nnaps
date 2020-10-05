import numpy as np
import tables
import warnings
import six
import yaml
import os

import pickle

import pandas as pd

from sklearn import preprocessing
from keras.models import model_from_json, model_from_yaml

# saveing to hdf5: https://gist.github.com/lukedeo/d1899f011ae41b26fb6e


'''
hacked out deepdish.io style keras NN saving functionality
[credit] deepdish
'''

# TODO: replace current saving of preprocessors by saving a pickeled version of them.




# Types that should be saved as pytables attribute
ATTR_TYPES = (int, float, bool, six.string_types,
              np.int8, np.int16, np.int32, np.int64,
              np.uint8, np.uint16, np.uint32, np.uint64,
              np.float16, np.float32, np.float64,
              np.bool_, np.complex64, np.complex128)

try:
    COMPRESSION = tables.Filters(complevel=9, complib='blosc', shuffle=True)
except Exception:
    warnings.warn("Missing BLOSC: no compression will be used.")
    COMPRESSION = tables.Filters()


def _save_level(handler, group, level, name=None, compress=True):
    if isinstance(level, dict):
        # First create a new group
        new_group = handler.create_group(group, name,
                                         "dict:{}".format(len(level)))
        for k, v in level.items():
            if isinstance(k, six.string_types):
                _save_level(handler, new_group, v, name=k)
            else:
                # Key is not string, so it gets a bit more complicated.
                # If the key is not a string, we will store it as a tuple instead,
                # inside a new group
                hsh = hash(k)
                if hsh < 0:
                    hname = 'm{}'.format(-hsh)
                else:
                    hname = '{}'.format(hsh)
                new_group2 = handler.create_group(new_group, '__pair_{}'.format(hname),
                                                  "keyvalue_pair")
                new_name = '__pair_{}'.format(hname)
                _save_level(handler, new_group2, k, name='key')
                _save_level(handler, new_group2, v, name='value')

                # new_name = '__keyvalue_pair_{}'.format(hash(name))
                # setattr(group._v_attrs, new_name, (name, level))
    elif isinstance(level, list):
        # Lists can contain other dictionaries and numpy arrays, so we don't
        # want to serialize them. Instead, we will store each entry as i0, i1,
        # etc.
        new_group = handler.create_group(group, name,
                                         "list:{}".format(len(level)))

        for i, entry in enumerate(level):
            level_name = 'i{}'.format(i)
            _save_level(handler, new_group, entry, name=level_name)

    elif isinstance(level, tuple):
        # Lists can contain other dictionaries and numpy arrays, so we don't
        # want to serialize them. Instead, we will store each entry as i0, i1,
        # etc.
        new_group = handler.create_group(group, name,
                                         "tuple:{}".format(len(level)))

        for i, entry in enumerate(level):
            level_name = 'i{}'.format(i)
            _save_level(handler, new_group, entry, name=level_name)

    elif isinstance(level, np.ndarray):
        atom = tables.Atom.from_dtype(level.dtype)
        if compress:
            node = handler.create_carray(group, name, atom=atom,
                                         shape=level.shape,
                                         chunkshape=level.shape,
                                         filters=COMPRESSION)
        else:
            node = handler.create_array(group, name, atom=atom,
                                        shape=level.shape)
        node[:] = level

    elif isinstance(level, ATTR_TYPES):
        setattr(group._v_attrs, name, level)

    elif level is None:
        # Store a None as an empty group
        new_group = handler.create_group(group, name, "nonetype:")

    else:
        warnings.warn('(deepdish.io.save) Pickling', level, ': '
                                                            'This may cause incompatiblities (for instance between '
                                                            'Python 2 and 3) and should ideally be avoided')
        node = handler.create_vlarray(group, name, tables.ObjectAtom())
        node.append(level)


def _load_level(level):
    if isinstance(level, tables.Group):
        dct = {}
        # Load sub-groups
        for grp in level:
            lev = _load_level(grp)
            n = grp._v_name
            # Check if it's a complicated pair or a string-value pair
            if n.startswith('__pair'):
                dct[lev['key']] = lev['value']
            else:
                dct[n] = lev

        # Load attributes
        for name in level._v_attrs._f_list():
            v = level._v_attrs[name]
            if isinstance(v, np.string_):
                v = v.decode('utf-8')
            dct[name] = v

        if level._v_title.startswith('list:'):
            N = int(level._v_title[len('list:'):])
            lst = []
            for i in range(N):
                lst.append(dct['i{}'.format(i)])
            return lst
        elif level._v_title.startswith('tuple:'):
            N = int(level._v_title[len('tuple:'):])
            lst = []
            for i in range(N):
                lst.append(dct['i{}'.format(i)])
            return tuple(lst)
        elif level._v_title.startswith('nonetype:'):
            return None
        else:
            return dct

    elif isinstance(level, tables.VLArray):
        if level.shape == (1,):
            return level[0]
        else:
            return level[:]
    elif isinstance(level, tables.Array):
        return level[:]


def save(path, data, compress=True):
    """
    Save any Python structure to an HDF5 file. It is particularly suited for
    Numpy arrays. This function works similar to ``numpy.save``, except if you
    save a Python object at the top level, you do not need to issue
    ``data.flat[1]`` to retrieve it from inside a Numpy array of type
    ``object``.
    Four types of objects get saved natively in HDF5, the rest get serialized
    automatically.  For most needs, you should be able to stick to the four,
    which are:
    * Dictionaries
    * Lists and tuples
    * Basic data types (including strings and None)
    * Numpy arrays
    A recommendation is to always convert your data to using only these four
    ingredients. That way your data will always be retrievable by any HDF5
    reader. A class that helps you with this is `deepdish.util.Saveable`.
    This function requires the [PyTables] module to be installed.
    Parameters
    ----------
    path : file-like object or string
        File or filename to which the data is saved.
    data : anything
        Data to be saved. This can be anything from a Numpy array, a string, an
        object, or a dictionary containing all of them including more
        dictionaries.
    compress : boolean
        Turn off data compression.
    See also
    --------
    load
    """
    if not isinstance(path, str):
        path = path.name

    h5file = tables.open_file(path, mode='w')

    # If the data is a dictionary, put it flatly in the root
    if isinstance(data, dict):
        group = h5file.root
        for key, value in data.items():
            _save_level(h5file, group, value, name=key, compress=compress)

    else:
        group = h5file.root
        _save_level(h5file, group, data, name='_top', compress=compress)
    h5file.close()


def load(path, unpack=False):
    """
    Loads an HDF5 saved with `save`.
    This function requires the [PyTables] module to be installed.
    Parameters
    ----------
    path : file-like object or string
        File or filename from which to load the data.
    unpack : bool
        If True, a single-entry dictionaries will be unpacked and the value
        will be returned directly. That is, if you save ``dict(a=100)``, only
        ``100`` will be loaded.
    Returns
    --------
    data : anything
        Hopefully an identical reconstruction of the data that was saved.
    See also
    --------
    save
    """
    if not isinstance(path, str):
        path = path.name

    h5file = tables.open_file(path, mode='r')
    root = h5file.root
    data = _load_level(h5file.root)
    # Unpack if top is the only one
    if isinstance(data, dict) and len(data) == 1:
        if '_top' in data:
            data = data['_top']
        elif unpack:
            data = data.values()[0]

    h5file.close()
    return data


def processors2dict(processors):
    # TODO: saving of categories_ for OneHot encoder should be adressed. Problem with saving unicode in hdf5.

    # TODO: processors2dict assumes that a processor only processes ONE feature and saves only ONE set of parameters

    processor_dict = {}

    for name, processor in processors.items():

        if processor.__class__ == preprocessing.OneHotEncoder:
            # deal with the categories_ attribute, which is an array of string of type object, but
            # hdf5 can't deal with that and has to convert this to byte format.

            categories_ = np.array(processor.categories_[0], dtype='S')
            dtype_ = np.array([processor.categories_[0].dtype], dtype='S')

            p = dict(preprocessor='OneHotEncoder',
                     kwargs={'categories_': categories_,
                             'categories_dtype': dtype_,
                             'drop_idx_': processor.drop_idx_})

        elif processor.__class__ == preprocessing.OrdinalEncoder:
            categories_ = np.array(processor.categories_[0], dtype='S')
            dtype_ = np.array([processor.categories_[0].dtype], dtype='S')

            p = dict(preprocessor='OrdinalEncoder',
                     kwargs={'categories_': categories_,
                             'categories_dtype': dtype_})

        elif processor.__class__ == preprocessing.StandardScaler:
            p = dict(preprocessor='StandardScaler',
                     kwargs={'scale_': processor.scale_, 'mean_': processor.mean_, 'var_': processor.var_,
                             'n_features_in_': processor.n_features_in_})

        elif processor.__class__ == preprocessing.RobustScaler:
            p = dict(preprocessor='RobustScaler',
                     kwargs={'scale_': processor.scale_, 'center_': processor.center_})

        elif processor.__class__ == preprocessing.MinMaxScaler:
            p = dict(preprocessor='MinMaxScaler',
                     kwargs={'scale_': processor.scale_, 'min_': processor.min_, 'data_min_': processor.data_min_,
                             'data_max_': processor.data_max_, 'data_range_': processor.data_range_})

        elif processor.__class__ == preprocessing.MaxAbsScaler:
            p = dict(preprocessor='MaxAbsScaler',
                     kwargs={'max_abs_': processor.max_abs_, 'scale_': processor.scale_})

        else:
            p = None

        processor_dict[name] = p

    return processor_dict


def dict2processors(processor_dict):
    # TODO: saving of categories_ for OneHot encoder should be addressed. Problem with saving unicode in hdf5.

    processors = {}

    for name, processor_data in processor_dict.items():

        if processor_data is None:
            # For this parameter, there is no preprocessor defined.
            p = None

        elif processor_data['preprocessor'] == 'OneHotEncoder':
            p = preprocessing.OneHotEncoder()
            # convert the categories_ attribute to the correct data type
            categories_ = processor_data['kwargs'].pop('categories_')
            dtype_ = processor_data['kwargs'].pop('categories_dtype')[0]
            processor_data['kwargs']['categories_'] = [np.array(np.array(categories_, dtype='U'), dtype=dtype_)]

        elif processor_data['preprocessor'] == 'OrdinalEncoder':
            p = preprocessing.OrdinalEncoder()
            # convert the categories_ attribute to the correct data type
            categories_ = processor_data['kwargs'].pop('categories_')
            dtype_ = processor_data['kwargs'].pop('categories_dtype')[0]
            processor_data['kwargs']['categories_'] = [np.array(np.array(categories_, dtype='U'), dtype=dtype_)]

        elif processor_data['preprocessor'] == 'StandardScaler':
            p = preprocessing.StandardScaler()

        elif processor_data['preprocessor'] == 'RobustScaler':
            p = preprocessing.RobustScaler()

        elif processor_data['preprocessor'] == 'MinMaxScaler':
            p = preprocessing.MinMaxScaler()

        elif processor_data['preprocessor'] == 'MaxAbsScaler':
            p = preprocessing.MaxAbsScaler()

        else:
            p = None

        if p is not None:
            for key, value in processor_data['kwargs'].items():
                setattr(p, key, value)

        processors[name] = p

    return processors


def convert_model(model):
    import codecs

    if type(model) == dict:
        # XGBoost model
        model_prep = model

    else:
        # KERAS model
        model_prep = {
        'config': model.to_yaml(),
        'weights': model.get_weights(),
        }

    return model_prep


def safe_model(model, processors, features, regressors, classifiers, setup, filename, history=None,
               train_data=None, test_data=None, method='hdf5'):

    # ignore user determined extension and replace by correct extention based on method.
    name, ext = os.path.splitext(filename)
    if method == 'hdf5':
        filename = name + '.h5'
    else:
        filename = name + '.dat'

    processor_dict = processors2dict(processors)
    model = convert_model(model)

    # print (type(model))
    # print('/n', model)

    setup = yaml.dump(setup)

    # features, regressors and classifiers have to be stored directly and their order is important.
    # the setup in yaml format does NOT keep the order.
    data = {
        'preprocessors': processor_dict,
        'setup': setup,
        'model': model,
        'features': features,
        'regressors': regressors,
        'classifiers': classifiers,
    }

    if history is not None:
        data['history_columns'] = list(history.columns)
        data['history'] = history.values

    if train_data is not None or test_data is not None:
        data['data'] = {}
        if train_data is not None:
            data['data']['train_data'] = train_data
        if test_data is not None:
            data['data']['test_data'] = test_data

    if method == 'hdf5':
        save(filename, data, compress=True)
    else:
        pickle.dump(data, open(filename, "wb"))


def load_model(filename):

    if os.path.splitext(filename)[1] == '.dat':
        data = pickle.load(open(filename, "rb"))
    else:
        data = load(filename)


    if 'config' in data['model'] and 'weights' in data['model']:
        model = model_from_yaml(data['model']['config'])
        W = data['model']['weights']
        model.set_weights(W)
    else:
        model = data['model']

    processors = dict2processors(data['preprocessors'])

    features = data.get('features', [])
    regressors = data.get('regressors', [])
    classifiers = data.get('classifiers', [])

    # load the setup and if necessary convert from yaml to a dictionary
    setup = data.get('setup', None)
    if setup is not None:
        setup = yaml.safe_load(setup)

    if 'history' in data:
        values = data['history']
        columns = data['history_columns']

        d = {}
        for i, column in enumerate(columns):
            d[column] = values[:,i]

        history = pd.DataFrame(data=d)
        history.index.name = 'epoch'
    else:
        history = None

    return model, processors, features, regressors, classifiers, setup, history
