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
                                   input_path_kw='path', input_path_prefix=base_path / 'test_data',
                                   star1_history_file='LOGS/history1.data', star2_history_file='LOGS/history2.data',
                                   binary_history_file='LOGS/binary_history.data', log_file='log.txt',
                                   output_path=base_path / 'test_data/hdf5')

            assert os.path.isfile(base_path / 'test_data/hdf5/M1.013_M0.331_P32.85_Z0.h5')
        finally:
            os.remove(base_path / 'test_data/hdf5/M1.013_M0.331_P32.85_Z0.h5')
            os.rmdir(base_path / 'test_data/hdf5/')

class TestExtract:

    def test_get_phases(self):
        phase_names = ['init', 'final', 'MLstart', 'MLend', 'ML', 'HeIgnition', 'HeCoreBurning', 'HeShellBurning']

        # stable model without He ignition
        data = extract_mesa.read_history(base_path / 'test_data/M1.022_M0.939_P198.55_Z0.h5')
        phases = extract_mesa.get_phases(data, phase_names)

        assert data['model_number'][phases['init']][0] == 3
        assert data['model_number'][phases['final']][0] == 11419
        assert data['model_number'][phases['MLstart']][0] == 2499
        assert data['model_number'][phases['MLend']][0] == 11106
        assert data['model_number'][phases['ML']][0] == 2499
        assert data['model_number'][phases['ML']][-1] == 11106
        assert phases['HeIgnition'] is None

        # CE model without He ignition
        data = extract_mesa.read_history(base_path / 'test_data/M1.239_M0.468_P165.41_Z0.h5')
        data = data[data['model_number'] <= 6198]
        phases = extract_mesa.get_phases(data, phase_names)

        assert data['model_number'][phases['ML']][0] == 2517
        assert data['model_number'][phases['ML']][-1] == 6198
        assert phases['HeIgnition'] is None
        assert phases['HeCoreBurning'] is None
        assert phases['HeShellBurning'] is None

        # stable model with degenerate He ignition
        data = extract_mesa.read_history(base_path / 'test_data/M0.840_M0.822_P554.20_Z0.h5')
        phases = extract_mesa.get_phases(data, phase_names)

        assert data['model_number'][phases['HeIgnition']][0] == 12453
        assert data['model_number'][phases['HeCoreBurning']][0] == 12267
        assert data['model_number'][phases['HeCoreBurning']][-1] == 13734
        assert data['model_number'][phases['HeShellBurning']][0] == 13737
        assert data['model_number'][phases['HeShellBurning']][-1] == 15087

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

        data = extract_mesa.read_history(base_path / 'test_data/M1.022_M0.939_P198.55_Z0.h5')

        parameters = ['star_1_mass__init', 'period_days__final', 'rl_1__max', 'rl_1__HeIgnition', 'age__ML__diff',
                      'he_core_mass__ML__rate']

        res = extract_mesa.extract_parameters(data, parameters)
        res = {k:v for k, v in zip(parameters, res)}

        assert res['star_1_mass__init'] == data['star_1_mass'][0]
        assert res['period_days__final'] == data['period_days'][-1]
        assert res['rl_1__max'] == np.max(data['rl_1'])
        assert np.isnan(res['rl_1__HeIgnition'])

        a1 = data['age'][data['lg_mstar_dot_1'] > -10][0]
        a2 = data['age'][(data['age'] > a1) & (data['lg_mstar_dot_1'] <= -10)][0]
        s = np.where((data['age'] >= a1) & (data['age'] <= a2))
        assert res['age__ML__diff'] == data['age'][s][-1] - data['age'][s][0]

        assert res['he_core_mass__ML__rate'] == (data['he_core_mass'][s][-1] - data['he_core_mass'][s][0]) / \
                                                (data['age'][s][-1] - data['age'][s][0])

    def test_is_stable(self):

        data = extract_mesa.read_history(base_path / 'test_data/M1.239_M0.468_P165.41_Z0.h5')

        stable, ce_age = extract_mesa.is_stable(data, criterion='Mdot', value=-3)
        assert stable is False
        assert ce_age == pytest.approx(5611615612.8, abs=0.1)

        stable, ce_age = extract_mesa.is_stable(data, criterion='delta', value=0.03)
        assert stable is False
        assert ce_age == pytest.approx(5611615629.7, abs=0.1)

        stable, ce_age = extract_mesa.is_stable(data, criterion='J_div_Jdot_div_P', value=10)
        assert stable is False
        assert ce_age == pytest.approx(5611615630.1, abs=0.1)

        stable, ce_age = extract_mesa.is_stable(data, criterion='M_div_Mdot_div_P', value=100)
        assert stable is False
        assert ce_age == pytest.approx(5611615628.8, abs=0.1)

        stable, ce_age = extract_mesa.is_stable(data, criterion='R_div_SMA', value=0.5)
        assert stable is False
        assert ce_age == pytest.approx(5611615618.8, abs=0.1)

    def test_apply_ce(self):

        data = extract_mesa.read_history(base_path / 'test_data/M1.235_M0.111_P111.58_Z0.h5')

        stable, ce_age = extract_mesa.is_stable(data, criterion='J_div_Jdot_div_P', value=10)
        data = data[data['age'] <= ce_age]

        data = extract_mesa.apply_ce(data, ce_model='')

        assert data['period_days'][-1] == pytest.approx(0.1085, abs=1e-4)
        assert data['binary_separation'][-1] == pytest.approx(0.0057, abs=1e-4)
        assert data['star_1_mass'][-1] == pytest.approx(0.3556, abs=1e-4)
        assert data['star_2_mass'][-1] == pytest.approx(0.1114, abs=1e-4)
        assert data['mass_ratio'][-1] == pytest.approx(3.1918, abs=1e-4)
        assert data['rl_1'][-1] == pytest.approx(0.5938, abs=1e-4)
        assert data['rl_2'][-1] == pytest.approx(0.3506, abs=1e-4)

    def test_extract_mesa(self):

        models = ['test_data/M1.235_M0.111_P111.58_Z0.h5',
                  'test_data/M1.239_M0.468_P165.41_Z0.h5',
                  'test_data/M1.022_M0.939_P198.55_Z0.h5',
                  'test_data/M0.840_M0.822_P554.20_Z0.h5',]
        models = pd.DataFrame(models, columns=['path'])

        parameters = ['star_1_mass__init',
                      'period_days__final',
                      'rl_1__max',
                      'rl_1__HeIgnition',
                      'age__ML__diff',
                      'he_core_mass__ML__rate',
                     ]

        results = extract_mesa.extract_mesa(models, stability_criterion='J_div_Jdot_div_P',
                                            stability_limit=10, parameters=parameters)

        # check dimensions and columns
        for p in parameters:
            assert p in results.columns
        assert 'stability' in results.columns
        assert len(results) == len(models)

        # check values of one of the models
        data = extract_mesa.read_history(base_path / 'test_data/M0.840_M0.822_P554.20_Z0.h5')
        result1 = extract_mesa.extract_parameters(data, parameters)
        result1 = {k: v for k, v in zip(parameters, result1)}

        result2 = results.loc[3]

        for k, v in result1.items():
            assert result2[k] == v
        assert result2['stability'] == 'stable'
