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

        data = extract_mesa.read_history(base_path / 'test_data/M1.022_M0.939_P198.55_Z0.h5')

        phases = ['init', 'final', 'MLstart', 'MLend', 'HeIgnition', 'HeCoreBurning']
        phases = extract_mesa.get_phases(data, phases)

        assert 1

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

