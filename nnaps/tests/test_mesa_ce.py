import pytest

from pathlib import Path

import numpy as np

from nnaps.mesa import extract_mesa, common_envelope

base_path = Path(__file__).parent

# by using scope='module', the predictor is created once and then used for all functions
# that need it in this test module.
@pytest.fixture(scope='module')
def data():
    data, _, profiles = extract_mesa.read_history(base_path / 'test_data/M1.080_M0.502_P192.67_Z0.01129.h5'
                                                  , return_profiles=True)

    stable, ce_age = common_envelope.is_stable(data, criterion='Mdot', value=-2)
    data = data[data['age'] <= ce_age]
    return data


def test_is_stable():
    data, _ = extract_mesa.read_history(base_path / 'test_data/M1.205_M0.413_P505.12_Z0.h5')

    stable, ce_age = common_envelope.is_stable(data, criterion='Mdot', value=-3)
    assert stable is False
    assert ce_age == pytest.approx(5179376595.6, abs=0.1)

    stable, ce_age = common_envelope.is_stable(data, criterion='delta', value=0.03)
    assert stable is False
    assert ce_age == pytest.approx(5179376616.3, abs=0.1)

    stable, ce_age = common_envelope.is_stable(data, criterion='J_div_Jdot_div_P', value=10)
    assert stable is False
    assert ce_age == pytest.approx(5179376617.0, abs=0.1)

    stable, ce_age = common_envelope.is_stable(data, criterion='M_div_Mdot_div_P', value=100)
    assert stable is False
    assert ce_age == pytest.approx(5179376614.8, abs=0.1)

    stable, ce_age = common_envelope.is_stable(data, criterion='R_div_SMA', value=0.5)
    assert stable is False
    assert ce_age == pytest.approx(5179376604.0, abs=0.1)


def test_apply_ce():
    data, _ = extract_mesa.read_history(base_path / 'test_data/M1.205_M0.413_P505.12_Z0.h5')

    stable, ce_age = common_envelope.is_stable(data, criterion='J_div_Jdot_div_P', value=10)
    data = data[data['age'] <= ce_age]

    data = common_envelope.apply_ce(data, ce_model='')

    assert data['period_days'][-1] == pytest.approx(25.55, abs=0.01)
    assert data['binary_separation'][-1] == pytest.approx(0.1775, abs=1e-4)
    assert data['star_1_mass'][-1] == pytest.approx(0.4477, abs=1e-4)
    assert data['star_2_mass'][-1] == pytest.approx(0.4278, abs=1e-4)
    assert data['mass_ratio'][-1] == pytest.approx(1.0465, abs=1e-4)
    assert data['rl_1'][-1] == pytest.approx(14.5993, abs=1e-4)
    assert data['rl_2'][-1] == pytest.approx(14.2991, abs=1e-4)


def test_apply_ce_profile(data):
    data, _, profiles = extract_mesa.read_history(base_path / 'test_data/M1.080_M0.502_P192.67_Z0.01129.h5'
                                                  , return_profiles=True)

    stable, ce_age = common_envelope.is_stable(data, criterion='Mdot', value=-2)
    data = data[data['age'] <= ce_age]

    profile = profiles['profile_1_mdot-2.0']

    af, M1_final = common_envelope.dewi_tauris2000(data, profile, a_th=0)

    assert af == pytest.approx(4.91, abs=0.01)
    assert M1_final == pytest.approx(0.383, abs=0.001)

    af, M1_final = common_envelope.dewi_tauris2000(data, profile, a_th=0.5)

    assert af == pytest.approx(6.463032, abs=0.000001)
    assert M1_final == pytest.approx(0.3859326, abs=0.000001)

# def test_all(data):
#
#     def period(af):
#         G = 2944.643655
#         P = np.sqrt(4 * np.pi ** 2 * af ** 3 / G * (M1_final + data['star_2_mass'][-1]))
#         return P
#
#     af, M1_final = common_envelope.iben_tutukov1984(data, al=1)
#
#     print('')
#     print('Iben & Tutukov 1984')
#     print('A final:', af)
#     print('P final:', period(af))
#     print('M1 final:', M1_final)
#
#     af, M1_final = common_envelope.webbink1984(data, al=1, lb=1)
#
#     print('')
#     print('Webbink 1984')
#     print('A final:', af)
#     print('P final:', period(af))
#     print('M1 final:', M1_final)
#
#     af, M1_final = common_envelope.demarco2011(data, al=1, lb=1)
#
#     print('')
#     print('De Marco 2011')
#     print('A final:', af)
#     print('P final:', period(af))
#     print('M1 final:', M1_final)
#
#     assert 0


def test_iben_tutukov(data):

    af, M1_final = common_envelope.iben_tutukov1984(data, al=1)

    assert M1_final == data['he_core_mass'][-1]
    assert af == pytest.approx(31.273, abs=0.001)


def test_webbink(data):

    af, M1_final = common_envelope.webbink1984(data, al=1, lb=1)

    assert M1_final == data['he_core_mass'][-1]
    assert af == pytest.approx(9.213, abs=0.001)


def test_demarco(data):

    af, M1_final = common_envelope.demarco2011(data, al=1, lb=1)

    assert M1_final == data['he_core_mass'][-1]
    assert af == pytest.approx(19.856, abs=0.001)
