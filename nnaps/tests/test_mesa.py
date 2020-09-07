import os

import pytest
import pandas as pd
import numpy as np
import pylab as pl

from nnaps.mesa import read_mesa, extract_mesa, evolution_phases, fileio

from pathlib import Path
base_path = Path(__file__).parent




class Test2H5:

    def test_read_mesa_output(self):

        print('basepath: ', base_path)

        filename = base_path / 'test_data/M1.013_M0.331_P32.85_Z0.00155/LOGS/history1.data'

        _, data = read_mesa.read_mesa_output(filename=filename, only_first=False)

        assert 'model_number' in data.dtype.names
        assert min(data['model_number']) == 1
        assert max(data['model_number']) == 30000
        assert len(data.dtype.names) == 53
        assert data.shape[0] == 10263


    def test_convert2hdf5(self):

        data = [[1.013, 0.331, 32.85, 0.12, -0.8, 0.00155, 749, 986, 0, 2000, 'M1.013_M0.331_P32.85_Z0.00155']]
        columns = ['M1MSun', 'M2MSun', 'PRLODays', 'PoverPMax', 'FeH', 'ZMIST', 'TtipMyr',
                   'GalAgeMyr', 'AgeBinNum', 'DeltaTBin', 'path']
        modellist = pd.DataFrame(data=data, columns=columns)

        try:
            read_mesa.convert2hdf5(modellist, star_columns=None, binary_columns=None, add_stopping_condition=True,
                                   skip_existing=False,
                                   input_path_kw='path', input_path_prefix=base_path / 'test_data',
                                   star1_history_file='LOGS/history1.data', star2_history_file='LOGS/history2.data',
                                   binary_history_file='LOGS/binary_history.data', log_file='log.txt',
                                   profile_files='all', profiles_path='LOGS', profile_pattern='profile_*.data',
                                   output_path=base_path / 'test_data/hdf5')

            assert os.path.isfile(base_path / 'test_data/hdf5/M1.013_M0.331_P32.85_Z0.00155.h5')
        finally:
            os.remove(base_path / 'test_data/hdf5/M1.013_M0.331_P32.85_Z0.00155.h5')
            os.rmdir(base_path / 'test_data/hdf5/')

class TestExtract:

    def test_get_phases(self):
        phase_names = ['init', 'final', 'MS', 'RGB', 'MLstart', 'MLend', 'ML', 'CE', 'CEstart', 'CEend', 'HeIgnition',
                       'HeCoreBurning', 'HeShellBurning']

        # test checking if all parameters are available.
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M0.789_M0.304_P20.58_Z0.h5', return_profiles=False)
        data = data[['age', 'period_days']]

        with pytest.raises(ValueError):
             phases = evolution_phases.get_all_phases(['MS'], data)

        # stable model without He ignition and struggles at the end
        # age of the last 1470 time steps doesn't change!
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M0.789_M0.304_P20.58_Z0.h5', return_profiles=False)
        phases = evolution_phases.get_all_phases(phase_names, data)

        assert data['model_number'][phases['init']][0] == 3
        assert data['model_number'][phases['final']][0] == 30000
        assert data['model_number'][phases['MS']][0] == 27
        assert data['model_number'][phases['MS']][-1] == 114
        assert data['model_number'][phases['RGB']][0] == 114
        assert data['model_number'][phases['RGB']][-1] == 948
        assert data['model_number'][phases['MLstart']][0] == 933
        assert data['model_number'][phases['MLend']][0] == 30000
        assert data['model_number'][phases['ML']][0] == 933
        assert data['model_number'][phases['ML']][-1] == 30000
        assert phases['HeIgnition'] is None

        # stable model without He ignition
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M0.814_M0.512_P260.18_Z0.h5', return_profiles=False)
        phases = evolution_phases.get_all_phases(phase_names, data)

        assert data['model_number'][phases['RGB']][0] == 111
        assert data['model_number'][phases['RGB']][-1] == 6570
        assert data['model_number'][phases['ML']][0] == 1290
        assert data['model_number'][phases['ML']][-1] == 7281
        assert phases['HeIgnition'] is None
        assert phases['HeCoreBurning'] is None
        assert phases['HeShellBurning'] is None

        # stable model with degenerate He ignition but issues in the He burning phase, and a double ML phase
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M1.125_M0.973_P428.86_Z0.h5', return_profiles=False)
        phases = evolution_phases.get_all_phases(phase_names, data)

        assert data['model_number'][phases['ML']][0] == 2556
        assert data['model_number'][phases['ML']][-1] == 19605
        assert data['model_number'][phases['HeIgnition']][0] == 19947
        assert phases['HeCoreBurning'] is None
        assert phases['HeShellBurning'] is None
        assert phases['CE'] is None
        assert phases['CEstart'] is None
        assert phases['CEend'] is None

        # CE model
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M1.205_M0.413_P505.12_Z0.h5', return_profiles=False)
        data = data[data['model_number'] <= 12111]
        data['CE_phase'][-1] = 1
        data['CE_phase'][-2] = 1
        phases = evolution_phases.get_all_phases(phase_names, data)

        assert data['model_number'][phases['ML']][0] == 2280
        assert data['model_number'][phases['ML']][-1] == 12111
        assert data['model_number'][phases['CE']][0] == 12108
        assert data['model_number'][phases['CE']][1] == 12111
        assert data['model_number'][phases['CEstart']][0] == 12108
        assert data['model_number'][phases['CEend']][0] == 12111
        assert phases['HeIgnition'] is None
        assert phases['HeCoreBurning'] is None
        assert phases['HeShellBurning'] is None

        # HB star with core and shell He burning
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M1.276_M1.140_P333.11_Z0.h5', return_profiles=False)
        phases = evolution_phases.get_all_phases(phase_names, data)

        assert data['model_number'][phases['ML']][0] == 2031
        assert data['model_number'][phases['ML']][-1] == 12018
        assert data['model_number'][phases['HeIgnition']][0] == 11709
        assert data['model_number'][phases['HeCoreBurning']][0] == 12492
        assert data['model_number'][phases['HeCoreBurning']][-1] == 12594
        assert data['model_number'][phases['HeShellBurning']][0] == 12597
        assert data['model_number'][phases['HeShellBurning']][-1] == 14268

        # sdB star with core and shell He burning
        data, _ = fileio.read_compressed_track(base_path / 'test_data/M1.269_M1.229_P133.46_Z0.00320.h5',
                                            return_profiles=False)
        phases = evolution_phases.get_all_phases(['sdA', 'sdB', 'sdO'], data)

        assert phases['sdA'] is None
        assert data['model_number'][phases['sdB']][0] == 22608
        assert data['model_number'][phases['sdB']][-1] == 22689
        assert phases['sdO'] is None

        a1, a2 = evolution_phases.HeCoreBurning(data, return_age=True)

        assert a1 == pytest.approx(3232213210.6798477, abs=0.001)
        assert a2 == pytest.approx(3316814816.4952917, abs=0.001)

    def test_decompose_parameter(self):

        pname, phase, func = evolution_phases.decompose_parameter('star_1_mass__init')
        assert pname == 'star_1_mass'
        assert phase == 'init'
        assert func.__name__ == 'avg_'

        pname, phase, func = evolution_phases.decompose_parameter('period_days__final')
        assert pname == 'period_days'
        assert phase == 'final'
        assert func.__name__ == 'avg_'

        pname, phase, func = evolution_phases.decompose_parameter('rl_1__max')
        assert pname == 'rl_1'
        assert phase is None
        assert func.__name__ == 'max_'

        pname, phase, func = evolution_phases.decompose_parameter('rl_1__HeIgnition')
        assert pname == 'rl_1'
        assert phase == 'HeIgnition'
        assert func.__name__ == 'avg_'

        pname, phase, func = evolution_phases.decompose_parameter('age__ML__diff')
        assert pname == 'age'
        assert phase == 'ML'
        assert func.__name__ == 'diff_'

        pname, phase, func = evolution_phases.decompose_parameter('he_core_mass__ML__rate')
        assert pname == 'he_core_mass'
        assert phase == 'ML'
        assert func.__name__ == 'rate_'

        pname, phase, func = evolution_phases.decompose_parameter('age__ML__diff')
        assert pname == 'age'
        assert phase == 'ML'
        assert func.__name__ == 'diff_'

        pname, phase, func = evolution_phases.decompose_parameter('star_1_mass__lg_mstar_dot_1_max')
        assert pname == 'star_1_mass'
        assert phase == 'lg_mstar_dot_1_max'
        assert func.__name__ == 'avg_'

    def test_extract_parameters(self):
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

        a1 = data['age'][data['lg_mstar_dot_1'] > -10][0]
        a2 = data['age'][(data['age'] > a1) & (data['lg_mstar_dot_1'] <= -10)][0]
        s = np.where((data['age'] >= a1) & (data['age'] <= a2))
        assert res['age__ML__diff'] == data['age'][s][-1] - data['age'][s][0]

        assert res['he_core_mass__ML__rate'] == (data['he_core_mass'][s][-1] - data['he_core_mass'][s][0]) / \
                                                (data['age'][s][-1] - data['age'][s][0])

        assert res['rl_1__HeIgnition'] == pytest.approx(152.8606, abs=0.0001)

        assert res['star_1_mass__lg_mstar_dot_1_max'] == pytest.approx(0.5205, abs=0.0001)

        phase_flags = ['ML', 'HeCoreBurning', 'He-WD']
        res = extract_mesa.extract_parameters(data, parameters, phase_flags=phase_flags)
        res = {k: v for k, v in zip(parameters+phase_flags, res)}

        assert res['ML'] is True
        assert res['HeCoreBurning'] is True
        assert res['He-WD'] is False

    def test_extract_mesa(self, root_dir):

        models = ['test_data/M0.789_M0.304_P20.58_Z0.h5',
                  'test_data/M0.814_M0.512_P260.18_Z0.h5',
                  'test_data/M1.276_M1.140_P333.11_Z0.h5'
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
        parameter_names = ['M1_init', 'P_final', 'rl_1__max', 'rl_1__HeIgnition', 'age__ML__diff', 'he_core_mass__ML__rate']

        phase_flags = ['sdB', 'He-WD']

        results = extract_mesa.extract_mesa(models, stability_criterion='J_div_Jdot_div_P', stability_limit=10,
                                            parameters=parameters, phase_flags=phase_flags,
                                            verbose=True)

        results.to_csv('test_results.csv', na_rep='nan')

        # check dimensions and columns
        for p in parameter_names:
            assert p in results.columns
        for p in phase_flags:
            assert p in results.columns
        assert 'stability' in results.columns
        assert len(results) == len(models)



