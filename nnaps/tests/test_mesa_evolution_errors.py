
import pytest

from nnaps.mesa import evolution_errors, fileio


def test_check_error_flags(base_path):
    # check no errors
    data, _ = fileio.read_compressed_track(base_path /
                                           'test_data/error_models/M0.851_M0.305_P358.92_Z0.00129_no_problems.h5')
    error_flags = evolution_errors.check_error_flags(data, 'log_g_upper_limit')
    assert error_flags == []

    # check max_model error
    error_flags = evolution_errors.check_error_flags(data, 'max_model_number')
    assert 1 in error_flags

    # check accretor overflow error
    error_flags = evolution_errors.check_error_flags(data, 'accretor_overflow_terminate')
    assert 2 in error_flags

    # check ML and He ignition error
    data, _ = fileio.read_compressed_track(base_path /
                                           'test_data/error_models/M2.407_M0.432_P1.72_Z0.00706_He_ignition_problem.h5')
    error_flags = evolution_errors.check_error_flags(data, '')
    assert 4 in error_flags

    # Check HeIgnition but no HeCoreBurning error
    data, _ = fileio.read_compressed_track(base_path /
                                           'test_data/error_models/M1.490_M1.380_P144.55_Z0.01427_Jz1.000_HeIgnition_but_no_Coreburning.h5')
    error_flags = evolution_errors.check_error_flags(data, '')
    assert 4 in error_flags

    # check CO core formation error
    data, _ = fileio.read_compressed_track(base_path /
                                           'test_data/error_models/M1.699_M1.401_P260.07_Z0.02489_Jz1.000_HeCoreBurning_doesnt_finish.h5')
    error_flags = evolution_errors.check_error_flags(data, '')
    assert 5 in error_flags