import pytest
import numpy as np

from nnaps.mesa.compress_mesa import  read_mesa_output

from pathlib import Path
base_path = Path(__file__).parent

class TestReadMesaOutput:

    def test_read_mesa_output_new_format(self):

        filename = base_path / 'test_data/history_mesa_v15140.data'

        model = read_mesa_output(filename=filename, only_first=False)

        assert type(model[0]['version_number'][0]) == np.str_
        assert model[0]['version_number'][0] == '15140'

        assert type(model[1]['model_number'][0]) == np.float_
        assert model[1]['model_number'][0] == 521