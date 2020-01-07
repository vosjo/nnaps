import unittest

from nnaps import defaults


class TestDefaultsGenerator(unittest.TestCase):

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
