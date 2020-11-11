import os
import copy
import pytest
import pandas as pd
import numpy as np
import pylab as pl

from nnaps.mesa import extract_mesa, evolution_phases, fileio


class TestProcessFileList:

    def test_process_file_list_parameters_included_in_filelist(self):

        file_list = pd.DataFrame(data={'path': ['path'] * 24,
                                       'stability_criterion': ['Mdot'] * 24,
                                       'stability_limit': np.random.uniform(-3, 0, 24),
                                       'ce_formalism': ['iben1984'] * 24,
                                       'ce_parameters': ["{'alpha_ce':0.3, 'alpha_th':0.3}"] * 24,
                                       'ce_profile_name': ['Mdot_profile'] * 24,
                                       })

        ol = copy.deepcopy(file_list)
        nl = extract_mesa.process_file_list(ol, stability_criterion='Mdot', stability_limit=-2,
                                            ce_formalism='Iben1984', ce_parameters={'alpha_ce': 0.5, 'alpha_th': 0.5},
                                            ce_profile_name='Mdot_profile')

        for par in ['path', 'stability_criterion', 'stability_limit', 'ce_formalism', 'ce_profile_name']:
            np.testing.assert_equal(file_list[par].values, nl[par].values)

        assert type(nl['ce_parameters'][0]) == dict, "ce_parameters should be converted to a dictionary"
        assert nl['ce_parameters'][0] == {'alpha_ce': 0.3, 'alpha_th': 0.3}, \
            "ce_parameters dictionary not correctly converted!"

    def test_process_file_list_parameters_not_included_in_filelist(self):

        file_list = pd.DataFrame(data={'path': ['path'] * 24,
                                       'stability_criterion': ['Mdot'] * 24,
                                       'stability_limit': np.random.uniform(-3, 0, 24),
                                       'ce_profile_name': ['Mdot_profile'] * 24,
                                       })

        ol = copy.deepcopy(file_list)
        nl = extract_mesa.process_file_list(ol, stability_criterion='Mdot', stability_limit=-2,
                                            ce_formalism='Iben1984', ce_parameters={'alpha_ce': 0.5, 'alpha_th': 0.5},
                                            ce_profile_name='Mdot_profile')

        for par in ['path', 'stability_criterion', 'stability_limit', 'ce_profile_name']:
            np.testing.assert_equal(file_list[par].values, nl[par].values)

        assert 'ce_formalism' in nl.columns.values
        np.testing.assert_equal(nl['ce_formalism'].values, np.array(['Iben1984'] * 24))

        assert 'ce_parameters' in nl.columns.values
        assert nl['ce_parameters'][0] == {'alpha_ce': 0.5, 'alpha_th': 0.5}


class TestExtract:

    def test_count_ml_phases(self, base_path):

        # 1 ML phase
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M0.789_M0.304_P20.58_Z0.h5')
        n_ml_phases = extract_mesa.count_ml_phases(data)
        assert n_ml_phases == 1

        # 2 separate ML phases
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M2.341_M1.782_P8.01_Z0.01412.h5')
        n_ml_phases = extract_mesa.count_ml_phases(data)
        assert n_ml_phases == 2

        # 1 ML phases and 3 other ML phases due to wind mass loss which should not get counted.
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M1.276_M1.140_P333.11_Z0.h5')
        n_ml_phases = extract_mesa.count_ml_phases(data)
        assert n_ml_phases == 1

    def test_extract_parameters(self, base_path):
        #TODO: improve this test case and add more checks

        # HB star with core and shell He burning
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M1.276_M1.140_P333.11_Z0.h5')

        parameters = ['star_1_mass__init', 'period_days__final', 'rl_1__max', 'rl_1__HeIgnition', 'age__ML__diff',
                      'he_core_mass__ML__rate', 'star_1_mass__lg_mstar_dot_1_max']

        res = extract_mesa.extract_parameters(data, parameters)
        res = {k:v for k, v in zip(parameters, res)}

        assert res['star_1_mass__init'] == data['star_1_mass'][0]
        assert res['period_days__final'] == data['period_days'][-1]
        assert res['rl_1__max'] == np.max(data['rl_1'])
        #assert np.isnan(res['rl_1__HeIgnition'])

        # a1 = data['age'][data['lg_mstar_dot_1'] > -10][0]
        # a2 = data['age'][(data['age'] > a1) & (data['lg_mstar_dot_1'] <= -10)][0]
        # s = np.where((data['age'] >= a1) & (data['age'] <= a2))
        # assert res['age__ML__diff'] == data['age'][s][-1] - data['age'][s][0]

        # assert res['he_core_mass__ML__rate'] == (data['he_core_mass'][s][-1] - data['he_core_mass'][s][0]) / \
        #                                         (data['age'][s][-1] - data['age'][s][0])

        assert res['rl_1__HeIgnition'] == pytest.approx(152.8606, abs=0.0001)

        assert res['star_1_mass__lg_mstar_dot_1_max'] == pytest.approx(0.5205, abs=0.0001)

        phase_flags = ['ML', 'HeCoreBurning', 'He-WD']
        res = extract_mesa.extract_parameters(data, parameters, phase_flags=phase_flags)
        res = {k: v for k, v in zip(parameters+phase_flags, res)}

        assert res['ML'] is True
        assert res['HeCoreBurning'] is True
        assert res['He-WD'] is False

    def test_extract_parameters_ml(self, base_path):
        """
        Test that the ML phases are correctly returned.

        n_ml_phases = 0:
            result = [<val>, <val>]

        n_ml_phases = 1:
            result = [<val>, [<val>]]

        n_ml_phases = 2:
            result = [<val>, [<val>, <val>]]
        """
        parameters = ['period_days__init', 'period_days__ML']

        data, _ = fileio.read_compressed_track(base_path / 'test_data/M2.341_M1.782_P8.01_Z0.01412.h5')

        result = extract_mesa.extract_parameters(data, parameters=parameters, phase_flags=[], n_ml_phases=0)
        assert len(result) == 2
        assert type(result[1]) != list

        result = extract_mesa.extract_parameters(data, parameters=parameters, phase_flags=[], n_ml_phases=1)
        assert len(result) == 2
        assert type(result[1]) == list
        assert len(result[1]) == 1

        result = extract_mesa.extract_parameters(data, parameters=parameters, phase_flags=[], n_ml_phases=2)
        assert len(result) == 2
        assert type(result[1]) == list
        assert len(result[1]) == 2


    def test_extract_mesa(self, root_dir):

            models = ['test_data/M0.789_M0.304_P20.58_Z0.h5',
                      'test_data/M0.814_M0.512_P260.18_Z0.h5',
                      'test_data/M1.276_M1.140_P333.11_Z0.h5',
                      'test_data/M2.341_M1.782_P8.01_Z0.01412.h5',
                      ]
            models = [os.path.join(root_dir, x) for x in models]

            models = pd.DataFrame(models, columns=['path'])

            parameters = [('star_1_mass__init', 'M1_init'),
                          ('period_days__final', 'P_final'),
                          'rl_1__max',
                          'rl_1__HeIgnition',
                          'age__ML__diff',
                          'he_core_mass__ML__rate',
                          ]
            parameter_names = ['M1_init', 'P_final', 'rl_1__max', 'rl_1__HeIgnition', 'age__ML__diff',
                               'he_core_mass__ML__rate']

            phase_flags = ['sdB', 'He-WD']

            results = extract_mesa.extract_mesa(models, stability_criterion='J_div_Jdot_div_P', stability_limit=10,
                                                parameters=parameters, phase_flags=phase_flags,
                                                verbose=True)

            # results.to_csv('test_results.csv', na_rep='nan')

            # check dimensions and columns
            for p in parameter_names:
                assert p in results.columns
            for p in phase_flags:
                assert p in results.columns
            for p in ['path', 'stability', 'n_ML_phases', 'error_flags']:
                assert p in results.columns
            assert len(results) == len(models)

            assert results['n_ML_phases'][0] == 1
            assert results['n_ML_phases'][3] == 2
