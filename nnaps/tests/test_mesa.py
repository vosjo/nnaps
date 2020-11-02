import os

import pytest
import pandas as pd

from nnaps.mesa import compress_mesa, fileio

from pathlib import Path
base_path = Path(__file__).parent


class Test2H5:

    def test_read_mesa_output(self):

        filename = base_path / 'test_data/M1.013_M0.331_P32.85_Z0.00155/LOGS/history1.data'

        _, data = compress_mesa.read_mesa_output(filename=filename, only_first=False)

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
            compress_mesa.convert2hdf5(modellist, star_columns=None, binary_columns=None, add_stopping_condition=True,
                                       skip_existing=False,
                                       input_path_kw='path', input_path_prefix=base_path / 'test_data',
                                       star1_history_file='LOGS/history1.data', star2_history_file='LOGS/history2.data',
                                       binary_history_file='LOGS/binary_history.data', log_file='log.txt',
                                       profile_files='all', profiles_path='LOGS', profile_pattern='profile_*.data',
                                       output_path=base_path / 'test_data/hdf5')

            assert os.path.isfile(base_path / 'test_data/hdf5/M1.013_M0.331_P32.85_Z0.00155.h5')

            data = fileio.read_hdf5(base_path / 'test_data/hdf5/M1.013_M0.331_P32.85_Z0.00155.h5')

            assert 'history' in data
            assert 'star1' in data['history']
            assert 'star2' in data['history']
            assert 'binary' in data['history']

            assert 'extra_info' in data
            assert 'nnaps-version' in data['extra_info']
            assert 'termination_code' in data['extra_info']

            assert 'profile_legend' in data
            assert 'profiles' in data

        finally:
            os.remove(base_path / 'test_data/hdf5/M1.013_M0.331_P32.85_Z0.00155.h5')
            os.rmdir(base_path / 'test_data/hdf5/')




