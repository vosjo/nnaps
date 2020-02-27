import os

import pytest
import pandas as pd
import numpy as np
import pylab as pl

from pathlib import Path
base_path = Path(__file__).parent

from nnaps.mesa import read_mesa, extract_mesa


class Test2H5:

    def test_read_mesa_output(self):

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

            assert os.path.isfile(base_path / 'test_data/hdf5/M1.013_M0.331_P32.85_Z0.h5')
        finally:
            os.remove(base_path / 'test_data/hdf5/M1.013_M0.331_P32.85_Z0.h5')
            os.rmdir(base_path / 'test_data/hdf5/')

class TestExtract:

    def test_get_phases(self):
        phase_names = ['init', 'final', 'MLstart', 'MLend', 'ML', 'HeIgnition', 'HeCoreBurning', 'HeShellBurning']

        # stable model without He ignition and struggles at the end
        # age of the last 1470 time steps doesn't change!
        data, _ = extract_mesa.read_history(base_path / 'test_data/M0.789_M0.304_P20.58_Z0.h5', return_profiles=False)
        phases = extract_mesa.get_phases(data, phase_names)

        assert data['model_number'][phases['init']][0] == 3
        assert data['model_number'][phases['final']][0] == 30000
        assert data['model_number'][phases['MLstart']][0] == 933
        assert data['model_number'][phases['MLend']][0] == 30000
        assert data['model_number'][phases['ML']][0] == 933
        assert data['model_number'][phases['ML']][-1] == 30000
        assert phases['HeIgnition'] is None

        # stable model without He ignition
        data, _ = extract_mesa.read_history(base_path / 'test_data/M0.814_M0.512_P260.18_Z0.h5', return_profiles=False)
        phases = extract_mesa.get_phases(data, phase_names)

        assert data['model_number'][phases['ML']][0] == 1290
        assert data['model_number'][phases['ML']][-1] == 7281
        assert phases['HeIgnition'] is None
        assert phases['HeCoreBurning'] is None
        assert phases['HeShellBurning'] is None

        # stable model with degenerate He ignition but issues in the He burning phase, and a double ML phase
        data, _ = extract_mesa.read_history(base_path / 'test_data/M1.125_M0.973_P428.86_Z0.h5', return_profiles=False)
        phases = extract_mesa.get_phases(data, phase_names)

        assert data['model_number'][phases['ML']][0] == 2556
        assert data['model_number'][phases['ML']][-1] == 19605
        assert data['model_number'][phases['HeIgnition']][0] == 19947
        assert phases['HeCoreBurning'] is None
        assert phases['HeShellBurning'] is None

        # CE model
        data, _ = extract_mesa.read_history(base_path / 'test_data/M1.205_M0.413_P505.12_Z0.h5', return_profiles=False)
        data = data[data['model_number'] <= 12111]
        phases = extract_mesa.get_phases(data, phase_names)

        assert data['model_number'][phases['ML']][0] == 2280
        assert data['model_number'][phases['ML']][-1] == 12111
        assert phases['HeIgnition'] is None
        assert phases['HeCoreBurning'] is None
        assert phases['HeShellBurning'] is None

        # HB star with core and shell He burning
        data, _ = extract_mesa.read_history(base_path / 'test_data/M1.276_M1.140_P333.11_Z0.h5', return_profiles=False)
        phases = extract_mesa.get_phases(data, phase_names)

        assert data['model_number'][phases['ML']][0] == 2031
        assert data['model_number'][phases['ML']][-1] == 12018
        assert data['model_number'][phases['HeIgnition']][0] == 11709
        assert data['model_number'][phases['HeCoreBurning']][0] == 11565
        assert data['model_number'][phases['HeCoreBurning']][-1] == 12594
        assert data['model_number'][phases['HeShellBurning']][0] == 12597
        assert data['model_number'][phases['HeShellBurning']][-1] == 14268

    def test_decompose_parameter(self):

        pname, phase, func = extract_mesa.decompose_parameter('star_1_mass__init')
        assert pname == 'star_1_mass'
        assert phase == 'init'
        assert func.__name__ == 'avg'

        pname, phase, func = extract_mesa.decompose_parameter('period_days__final')
        assert pname == 'period_days'
        assert phase == 'final'
        assert func.__name__ == 'avg'

        pname, phase, func = extract_mesa.decompose_parameter('rl_1__max')
        assert pname == 'rl_1'
        assert phase is None
        assert func.__name__ == 'max'

        pname, phase, func = extract_mesa.decompose_parameter('rl_1__HeIgnition')
        assert pname == 'rl_1'
        assert phase == 'HeIgnition'
        assert func.__name__ == 'avg'

        pname, phase, func = extract_mesa.decompose_parameter('age__ML__diff')
        assert pname == 'age'
        assert phase == 'ML'
        assert func.__name__ == 'diff'

        pname, phase, func = extract_mesa.decompose_parameter('he_core_mass__ML__rate')
        assert pname == 'he_core_mass'
        assert phase == 'ML'
        assert func.__name__ == 'rate'

        pname, phase, func = extract_mesa.decompose_parameter('age__ML__diff')
        assert pname == 'age'
        assert phase == 'ML'
        assert func.__name__ == 'diff'

    def test_extract_parameters(self):
        #TODO: improve this test case and add more checks

        # HB star with core and shell He burning
        data, _ = extract_mesa.read_history(base_path / 'test_data/M1.276_M1.140_P333.11_Z0.h5')

        parameters = ['star_1_mass__init', 'period_days__final', 'rl_1__max', 'rl_1__HeIgnition', 'age__ML__diff',
                      'he_core_mass__ML__rate']

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

        phase_flags = ['ML', 'HeCoreBurning']
        res = extract_mesa.extract_parameters(data, parameters, phase_flags=phase_flags)
        res = {k: v for k, v in zip(parameters+phase_flags, res)}

        assert res['ML'] is True
        assert res['HeCoreBurning'] is True

    def test_is_stable(self):

        data, _ = extract_mesa.read_history(base_path / 'test_data/M1.205_M0.413_P505.12_Z0.h5')

        stable, ce_age = extract_mesa.is_stable(data, criterion='Mdot', value=-3)
        assert stable is False
        assert ce_age == pytest.approx(5179376595.6, abs=0.1)

        stable, ce_age = extract_mesa.is_stable(data, criterion='delta', value=0.03)
        assert stable is False
        assert ce_age == pytest.approx(5179376616.3, abs=0.1)

        stable, ce_age = extract_mesa.is_stable(data, criterion='J_div_Jdot_div_P', value=10)
        assert stable is False
        assert ce_age == pytest.approx(5179376617.0, abs=0.1)

        stable, ce_age = extract_mesa.is_stable(data, criterion='M_div_Mdot_div_P', value=100)
        assert stable is False
        assert ce_age == pytest.approx(5179376614.8, abs=0.1)

        stable, ce_age = extract_mesa.is_stable(data, criterion='R_div_SMA', value=0.5)
        assert stable is False
        assert ce_age == pytest.approx(5179376604.0, abs=0.1)

    def test_apply_ce(self):

        data, _ = extract_mesa.read_history(base_path / 'test_data/M1.205_M0.413_P505.12_Z0.h5')

        stable, ce_age = extract_mesa.is_stable(data, criterion='J_div_Jdot_div_P', value=10)
        data = data[data['age'] <= ce_age]

        data = extract_mesa.apply_ce(data, ce_model='')

        assert data['period_days'][-1] == pytest.approx(25.55, abs=0.01)
        assert data['binary_separation'][-1] == pytest.approx(0.1775, abs=1e-4)
        assert data['star_1_mass'][-1] == pytest.approx(0.4477, abs=1e-4)
        assert data['star_2_mass'][-1] == pytest.approx(0.4278, abs=1e-4)
        assert data['mass_ratio'][-1] == pytest.approx(1.0465, abs=1e-4)
        assert data['rl_1'][-1] == pytest.approx(14.5993, abs=1e-4)
        assert data['rl_2'][-1] == pytest.approx(14.2991, abs=1e-4)

    def test_extract_mesa(self):

        models = ['test_data/M0.789_M0.304_P20.58_Z0.h5',
                  'test_data/M0.814_M0.512_P260.18_Z0.h5',
                  ]
        models = pd.DataFrame(models, columns=['path'])

        parameters = ['star_1_mass__init',
                      'period_days__final',
                      'rl_1__max',
                      'rl_1__HeIgnition',
                      'age__ML__diff',
                      'he_core_mass__ML__rate',
                     ]

        results = extract_mesa.extract_mesa(models, stability_criterion='J_div_Jdot_div_P',
                                            stability_limit=10, parameters=parameters, verbose=True)

        # check dimensions and columns
        for p in parameters:
            assert p in results.columns
        assert 'stability' in results.columns
        assert len(results) == len(models)


