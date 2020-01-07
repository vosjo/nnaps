import unittest

from nnaps import defaults


class TestDefaultsGenerator(unittest.TestCase):

    def test_get_processor_class(self):

        self.assertEqual(defaults.get_processor_class('OneHotEncoder')().__class__.__name__,
                         'OneHotEncoder')

        self.assertEqual(defaults.get_processor_class('StandardScaler')().__class__.__name__,
                         'StandardScaler')

        self.assertEqual(defaults.get_processor_class('RobustScaler')().__class__.__name__,
                         'RobustScaler')

        self.assertEqual(defaults.get_processor_class('MinMaxScaler')().__class__.__name__,
                         'MinMaxScaler')

        self.assertEqual(defaults.get_processor_class('MaxAbsScaler')().__class__.__name__,
                         'MaxAbsScaler')

        self.assertEqual(defaults.get_processor_class('FakeEncoder'), None)


    def test_add_defaults_to_setup(self):
        setup = {'features': ['M1', 'qinit', 'Pinit', 'FeHinit'],
                 'regressors': ['Pfinal', 'qfinal'],
                 'classifiers': ['product', 'binary_type'], }

        setup_new = defaults.add_defaults_to_setup(setup)

        self.assertTrue('processor' in setup_new['features']['M1'])
        self.assertEqual(setup_new['features']['M1']['processor'], defaults.default_scaler)

        self.assertTrue('processor' in setup_new['regressors']['Pfinal'])
        self.assertEqual(setup_new['regressors']['Pfinal']['processor'], None)

        self.assertTrue('processor' in setup_new['classifiers']['product'])
        self.assertEqual(setup_new['classifiers']['product']['processor'], defaults.default_encoder)


    def test_default_preprocessor_update(self):

        setup = {'features':  {'M1': {'processor': 'StandardScaler'},
                               'qinit': {'processor': 'RobustScaler'},
                               'Pinit': {'processor': 'MinMaxScaler'},
                               'FeHinit': {'processor': 'MaxAbsScaler'},},

                 'regressors': ['Pfinal', 'qfinal'],

                 'classifiers': {'binary_type': {'processor': None},
                                 'product': {'processor': 'OneHotEncoder'},},
                 }

        setup_new = defaults.add_defaults_to_setup(setup)

        for key in setup['features'].keys():
            self.assertEqual(setup_new['features'][key]['processor'].__name__, setup['features'][key]['processor'],
                             msg="{} does not have preprocessor with correct class.".format(key) +
                                 " got {}, expected {}".format(setup_new['features'][key]['processor'].__name__,
                                                              setup['features'][key]['processor']))

        for key in setup['regressors']:
            self.assertEqual(setup_new['regressors'][key]['processor'], None,
                             msg="{} should not have a preprocessor.".format(key) +
                                 " got {}, expected {}".format(setup_new['regressors'][key]['processor'], None))

        self.assertEqual(setup_new['classifiers']['binary_type']['processor'], None,
                         msg="{} should not have a preprocessor.".format('binary_type') +
                             " got {}, expected {}".format(setup_new['classifiers']['binary_type']['processor'], None))

        self.assertEqual(setup_new['classifiers']['product']['processor'].__name__, 'OneHotEncoder',
                         msg="{} does not have preprocessor with correct class.".format('product') +
                             " got {}, expected OneHotEncoder".format(setup_new['classifiers']['product']['processor'].__name__))