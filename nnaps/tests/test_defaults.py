import pytest

from nnaps import defaults


class TestDefaultsGenerator:

    def test_get_processor_class(self):

        assert defaults.get_processor_class('OneHotEncoder')().__class__.__name__ == 'OneHotEncoder'

        assert defaults.get_processor_class('StandardScaler')().__class__.__name__ == 'StandardScaler'

        assert defaults.get_processor_class('RobustScaler')().__class__.__name__ == 'RobustScaler'

        assert defaults.get_processor_class('MinMaxScaler')().__class__.__name__ == 'MinMaxScaler'

        assert defaults.get_processor_class('MaxAbsScaler')().__class__.__name__ == 'MaxAbsScaler'

        assert defaults.get_processor_class('FakeEncoder') is None

    def test_add_defaults_to_setup(self):
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

    def test_default_preprocessor_update(self):

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
            assert setup_new['features'][key]['processor'].__name__ == setup['features'][key]['processor']

        for key in setup['regressors']:
            assert setup_new['regressors'][key]['processor'] is None

        assert setup_new['classifiers']['binary_type']['processor'] is None

        assert setup_new['classifiers']['product']['processor'].__name__ == 'OneHotEncoder'
