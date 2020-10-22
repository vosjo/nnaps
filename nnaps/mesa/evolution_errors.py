
import numpy as np

from nnaps.mesa.evolution_phases import _check_history_parameters, HeIgF


def mass_loss_error(history):
    """
    Check if there is a possible issue with mass loss. This function checks if RLOF is still ongoing when the model
    stops. For a stable model this would be an indication that something is wrong.

    Required history parameters:
        - lg_mstar_dot_1
        - lg_wind_mdot_1

    :param history: the evolution history of the system.
    :type history: numpy ndarray
    :return: True if there is a mass loss error, else False
    """

    required_parameters = ['lg_mstar_dot_1', 'lg_wind_mdot_1']
    if not _check_history_parameters(history, required_parameters, evol_phase='ML error', raise_error=False):
        return False

    mass_loss = np.log10(10 ** history['lg_mstar_dot_1'] - 10 ** history['lg_wind_mdot_1'])

    return mass_loss[-1] > -10


def he_ignition_error(history):
    """
    Check if there are issues with He ignition. The function checks if there is ignition but no actual core He
    burning, and also if there is a ramp up to ignition, but the ignition itself is unsuccessful.

    Required history parameters:
        - age
        - log_LHe
        - log_center_T
        - log_center_Rho

    :param history: the evolution history of the system.
    :type history: numpy ndarray
    :return: True if there is a He ignition error, else False
    """
    required_parameters = ['age', 'log_LHe', 'log_center_T', 'log_center_Rho']
    if not _check_history_parameters(history, required_parameters, evol_phase='He ignition error', raise_error=False):
        return False

    # Check if there is a ramp-up to He ignition, but no actual ignition. Check the last 2000 points
    if np.all(history['log_center_T'] < HeIgF(history['log_center_Rho'])) and np.all(history['log_LHe'] < 1):
        d = history[history['age'] > history[history['log_LHe'] > -50]['age'][0]]
        if len(d) > 5000:
            log_LHe = d['log_LHe'][-5000:]
            log_LHe_avg = np.average(log_LHe)
            if (log_LHe > log_LHe_avg - 0.5).all() and (log_LHe < log_LHe_avg + 0.5).all() \
                    and log_LHe_avg > -50 and log_LHe_avg > history['log_LHe'][0]:
                return True

    # Check if there is He ignition but no core burning
    if np.any(history['log_LHe'] > 1) and np.all(history['log_center_T'] < HeIgF(history['log_center_Rho'])):
        return True

    return False


def he_core_burning_error(history):
    """
    Check if there is an issue with He core burning. The function checks if core He burning starts, but never ends
    before the end of the evolution.

    Required history parameters:
        - c_core_mass
        - log_center_T
        - log_center_Rho

    :param history: the evolution history of the system.
    :type history: numpy ndarray
    :return: True if there is a He core burning error, else False
    """
    required_parameters = ['c_core_mass', 'log_center_T', 'log_center_Rho']
    if not _check_history_parameters(history, required_parameters, evol_phase='He ignition error', raise_error=False):
        return False

    if np.any(history['log_center_T'] >= HeIgF(history['log_center_Rho'])) and np.all(history['c_core_mass'] < 0.01):
        return True
    else:
        return False


def check_error_flags(history, termination_code):
    """
    Check for some possible errors in the model and report them.

    Errors that are checked and there codes:

        1. MESA stopped because of max model number
        2. MESA stopped because the accretor is overflowing it's Roche-lobe
        3. The mass loss phase doesn't end, to check if mass loss didn't cause mesa to crash:
           :func:`~nnaps.mesa.evolution_errors.mass_loss_error`
        4. Potential problem with He ignition: He ramps up and doesnt ignite, or He ignites but there is no core
           burning: :func:`~nnaps.mesa.evolution_errors.he_ignition_error`
        5. He core burning starts, but there is no formation of a CO core:
           :func:`~nnaps.mesa.evolution_errors.he_core_burning_error`


    :param history: the evolution history of the system
    :type history: numpy ndarray
    :param termination_code: the termination code of the mesa model
    :type termination_code: str
    :return: list of integers with error codes
    """

    error_codes = []

    # Check termination_code
    if termination_code == 'max_model_number':
        error_codes.append(1)
    if termination_code == 'accretor_overflow_terminate':
        error_codes.append(2)

    # Check Mass loss error
    if mass_loss_error(history):
        error_codes.append(3)

    # Check He ignition error
    if he_ignition_error(history):
        error_codes.append(4)

    # check He core burning errors
    if he_core_burning_error(history):
        error_codes.append(5)

    return error_codes
