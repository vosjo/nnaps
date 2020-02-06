import os

import pytest
import pandas as pd

from pathlib import Path
base_path = Path(__file__).parent

from nnaps.mesa import read_mesa


def test_read_mesa_output():

    filename = base_path / 'test_data/M1.013_M0.331_P32.85_Z0.00155/LOGS/history1.data'

    _, data = read_mesa.read_mesa_output(filename=filename, only_first=False)

    assert 'model_number' in data.dtype.names
    assert min(data['model_number']) == 1
    assert max(data['model_number']) == 30000
    assert len(data.dtype.names) == 53
    assert data.shape[0] == 10263


def test_convert2hdf5():

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