import pytest

from keras.optimizers import  Optimizer

from nnaps import defaults


def test_update_with_default_dict():
    default_dict = {'processor': 'StandardScaler', 'loss': 'mae'}

    #-- usecase where the level2 value is a list
    setup = {'features': ['M1', 'qinit', 'Pinit', 'FeHinit'],
             'regressors': ['Pfinal', 'qfinal'],
             'classifiers': ['product', 'binary_type'], }

    setup_new = defaults.update_with_default_dict(setup, 'regressors', default_dict)

    assert 'processor' in setup_new['regressors']['Pfinal']
    assert setup_new['regressors']['Pfinal']['processor'] == default_dict['processor']
    assert 'loss' in setup_new['regressors']['Pfinal']
    assert setup_new['regressors']['Pfinal']['loss'] == default_dict['loss']

    #-- same usecase but with inplace updates
    setup = {'features': ['M1', 'qinit', 'Pinit', 'FeHinit'],
             'regressors': ['Pfinal', 'qfinal'],
             'classifiers': ['product', 'binary_type'], }

    defaults.update_with_default_dict(setup, 'regressors', default_dict, in_place=True)

    assert setup['regressors'] != ['Pfinal', 'qfinal'], "No in-place update was done, setup['regressors'] " + \
                                                        "should return a dictionary not: {}".format(setup['regressors'])
    assert 'processor' in setup['regressors']['Pfinal']
    assert setup['regressors']['Pfinal']['processor'] == default_dict['processor']
    assert 'loss' in setup['regressors']['Pfinal']
    assert setup['regressors']['Pfinal']['loss'] == default_dict['loss']


    #-- usecase where the level2 value is a dictionary
    setup = {'features': ['M1', 'qinit', 'Pinit', 'FeHinit'],

             'regressors': {'Pfinal': None,
                            'qfinal': {'processor': 'StandardScaler', 'loss': 'mafe'},
                            'M1final': {'processor': 'RobustScaler'},
                            },

             'classifiers': ['product', 'binary_type']
             }

    setup_new = defaults.update_with_default_dict(setup, 'regressors', default_dict)

    print (setup_new)

    # Pfinal didn't contain any info and should be replaced by default values
    assert setup_new['regressors']['Pfinal'] == default_dict

    # qfinal contains both values, and should not be kept as is
    assert setup_new['regressors']['qfinal'] == {'processor': 'StandardScaler', 'loss': 'mafe'}

    # M1final contained one none default value that should be kept, and loss should be added
    assert setup_new['regressors']['M1final'] == {'processor': 'RobustScaler', 'loss': 'mae'}

    # check that no in-place update was done
    assert setup_new['regressors'] != setup['regressors']

def test_get_optimizer():

    opt = defaults.get_optimizer('adam', optimizer_kwargs=None)

    assert issubclass(opt.__class__, Optimizer)

    with pytest.raises(ValueError):
        opt = defaults.get_optimizer('fake', optimizer_kwargs=None)

def test_get_processor_class():

    def test_processor(name):
        # for a processor to be valid it needs to have 4 methods: fit, transform, fit_transform and inverse_transform
        proc = defaults.get_processor_class(name)
        assert hasattr(proc, 'fit')
        assert hasattr(proc, 'transform')
        assert hasattr(proc, 'fit_transform')
        assert hasattr(proc, 'inverse_transform')

    test_processor('OneHotEncoder')
    test_processor('StandardScaler')
    test_processor('RobustScaler')
    test_processor('MinMaxScaler')
    test_processor('MaxAbsScaler')

    with pytest.raises(ValueError):
        proc = defaults.get_processor_class('FakeEncoder')


class TestAddingDefaults:

    def test_set_default_scaler_encoder(self):

        setup = {
                 'default_scaler': 'MinMaxScaler', 'default_encoder':'OrdinalEncoder',
                 'features': ['M1', 'qinit', 'Pinit', 'FeHinit'],
                 'regressors': ['Pfinal', 'qfinal'],
                 'classifiers': ['product', 'binary_type'],
                 }

        setup_new = defaults.add_defaults_to_setup(setup)

        assert setup_new['features']['M1']['processor'] == 'MinMaxScaler'
        assert setup_new['classifiers']['product']['processor'] == 'OrdinalEncoder'

    def test_add_processor_defaults_to_setup(self):

        #-- case where features, regressors and classifiers are lists
        setup = {'features': ['M1', 'qinit', 'Pinit', 'FeHinit'],
                 'regressors': ['Pfinal', 'qfinal'],
                 'classifiers': ['product', 'binary_type'], }

        setup_new = defaults.add_defaults_to_setup(setup)

        assert 'processor' in setup_new['features']['M1']
        assert setup_new['features']['M1']['processor'] == defaults.default_scaler

        assert 'processor' in setup_new['regressors']['Pfinal']
        assert setup_new['regressors']['Pfinal']['processor'] is None

        assert 'processor' in setup_new['classifiers']['product']
        assert setup_new['classifiers']['product']['processor'] == defaults.default_encoder

        #-- case where features, regressors and classifiers are dictionaries

        setup = {'features':  {'M1': {'processor': 'StandardScaler'},
                               'qinit': {'processor': 'RobustScaler'},
                               'Pinit': {'processor': 'MinMaxScaler'},
                               'FeHinit': {'processor': 'MaxAbsScaler'}, },

                 'regressors': ['Pfinal', 'qfinal'],

                 'classifiers': {'binary_type': {'processor': None},
                                 'product': None, },
                 }

        setup_new = defaults.add_defaults_to_setup(setup)

        for key in setup['features'].keys():
            assert setup_new['features'][key]['processor'] == setup['features'][key]['processor']

        for key in setup['regressors']:
            assert setup_new['regressors'][key]['processor'] is None

        assert setup_new['classifiers']['binary_type']['processor'] is None

        assert setup_new['classifiers']['product']['processor'] == 'OneHotEncoder'

    def test_add_loss_defaults_to_setup(self):

        #-- case where features, regressors and classifiers are lists
        setup = {'features': ['M1', 'qinit', 'Pinit', 'FeHinit'],
                 'regressors': ['Pfinal', 'qfinal'],
                 'classifiers': ['product', 'binary_type'], }

        setup_new = defaults.add_defaults_to_setup(setup)

        assert 'loss' in setup_new['regressors']['Pfinal']
        assert setup_new['regressors']['Pfinal']['loss'] == defaults.default_regressor_loss

        assert 'loss' in setup_new['classifiers']['product']
        assert setup_new['classifiers']['product']['loss'] == defaults.default_classifier_loss

        #-- case where features, regressors and classifiers are dictionaries

        setup = {'features':  {'M1': {'processor': 'StandardScaler'},
                               'qinit': {'processor': 'RobustScaler'},
                               'Pinit': {'processor': 'MinMaxScaler'},
                               'FeHinit': {'processor': 'MaxAbsScaler'}, },

                 'regressors': {'Pfinal': {'processor': 'StandardScaler', 'loss': 'mafe'},
                                'qfinal': {'processor': 'RobustScaler'},
                               },

                 'classifiers': {'product': None, },
                 }

        setup_new = defaults.add_defaults_to_setup(setup)

        assert 'loss' in setup_new['regressors']['Pfinal']
        assert setup_new['regressors']['Pfinal']['loss'] == 'mafe'

        assert setup_new['regressors']['qfinal']['loss'] == defaults.default_regressor_loss

        assert 'loss' in setup_new['classifiers']['product']
        assert setup_new['classifiers']['product']['loss'] == defaults.default_classifier_loss

    def test_add_defaults_to_setup(self):

        setup = {'features': ['M1', 'qinit', 'Pinit', 'FeHinit'],
                 'regressors': ['Pfinal', 'qfinal'],
                 'classifiers': ['product', 'binary_type'], }

        setup_new = defaults.add_defaults_to_setup(setup)

        assert 'model' in setup_new

        assert 'random_state' in setup_new
        assert 'train_test_split' in setup_new

        assert 'optimizer' in setup_new
        assert 'optimizer_kwargs' in setup_new