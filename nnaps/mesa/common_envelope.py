
import numpy as np
from numpy.lib.recfunctions import append_fields

STABILITY_CRITERIA = ['Mdot', 'delta', 'J_div_Jdot_div_P', 'M_div_Mdot_div_P', 'R_div_SMA']
CE_FORMALISMS = ['iben_tutukov1984', 'webbink1984', 'dewi_tauris2000', 'demarco2011']

def is_stable(data, criterion='J_div_Jdot_div_P', value=10, return_model_number=False):
    """
    Checks if a model is stable with respect to the provided stability criterion. Known criteria are:

    - Mdot: lg_mstar_dot_1 > value
    - delta: mass_transfer_delta > value
    - J_div_Jdot_div_P: 10**log10_J_div_Jdot_div_P < value
    - M_div_Mdot_div_P: 10**log10_M_div_Mdot_div_P < value
    - R_div_SMA: star_1_radius / binary_separation > value

    :param data: model (np ndarray)
    :param criterion: which criterion to use
    :param value: value for the criterion to satisfy
    :param return_model_number: if true also return the model_number
    :return: stable (boolean), age (float), [model_number (int)], when stable age and model_number are last reached values
             when unstable, they are the age and model_number when the model becomes unstable.
    """
    stable = True
    a = data['age'][-1]
    m = data['model_number'][-1]

    if criterion not in STABILITY_CRITERIA:
        raise ValueError('Stability criterion not recognized. Use any of: ' + str(STABILITY_CRITERIA))

    if criterion == 'Mdot':

        if np.max(data['lg_mstar_dot_1']) > value:
            s = np.where(data['lg_mstar_dot_1'] > value)
            stable = False

    elif criterion == 'delta':

        if np.max(data['mass_transfer_delta']) > value:
            s = np.where(data['mass_transfer_delta'] > value)
            stable = False

    elif criterion == 'J_div_Jdot_div_P':

        if np.min(10 ** data['log10_J_div_Jdot_div_P']) <= value:
            s = np.where(10 ** data['log10_J_div_Jdot_div_P'] < value)
            stable = False

    elif criterion == 'M_div_Mdot_div_P':

        if np.min(10 ** data['log10_M_div_Mdot_div_P']) <= value:
            s = np.where(10 ** data['log10_M_div_Mdot_div_P'] < value)
            stable = False

    elif criterion == 'R_div_SMA':

        if np.max(data['star_1_radius'] / data['binary_separation']) > value:
            s = np.where(data['star_1_radius'] / data['binary_separation'] > value)
            stable = False


    if not stable:
        a = data['age'][s][0]
        m = data['model_number'][s][0]

    if return_model_number:
        return stable, a, m
    else:
        return stable, a


def apply_ce(data, profiles=None, ce_formalism='iben_tutukov1984', max_profile_distance=5, **kwargs):
    """
    Function performs the ce ejection and updates some stellar and binary parameters

    Different CE formalisms are supported:

    - iben_tutukov1984:
      `Iben & Tutukov 1984, ApJ, 284, 719 <https://ui.adsabs.harvard.edu/abs/1984ApJ...284..719I/abstract>`_
    - webbink1984:
      `Webbink 1984, ApJ, 277, 355 <https://ui.adsabs.harvard.edu/abs/1984ApJ...277..355W/abstract>`_
    - dewi_tauris2000:
      `Dewi and Tauris 2000, A&A, 360, 1043 <https://ui.adsabs.harvard.edu/abs/2000A%26A...360.1043D/abstract>`_
    - demarco2011:
      `De Marco et al. 2011, MNRAS, 411, 2277 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.411.2277D/abstract>`_

    for more details on each of the formalisms and which parameters are required, see their respective functions below.

    updates when available:
        - star_1_mass
        - envelope_mass
        - period_days
        - binary_separation
        - mass_ratio
        - rl_1
        - rl_2

    :param data: ndarray with model parameters
    :param profiles: dictionary containing all available profiles for the model
    :param ce_formalism: Which CE formalism to use
    :param max_profile_distance: when using dewi_tauris2000, the maximum model_number difference between onset CE and
           the closest profile available.
    :return: same dataset as provided with on the last line the parameters after the CE phase.
    """

    if ce_formalism not in CE_FORMALISMS:
        raise ValueError('CE formalism not recognized, use one of: ' + str(CE_FORMALISMS))

    if ce_formalism == 'iben_tutukov1984':
        af, M1_final = iben_tutukov1984(data, **kwargs)

    elif ce_formalism == 'webbink1984':
        af, M1_final = webbink1984(data, **kwargs)

    elif ce_formalism == 'demarco2011':
        af, M1_final = demarco2011(data, **kwargs)

    elif ce_formalism == 'dewi_tauris2000':
        # need to find the correct profile for this

        if type(profiles) is not dict:
            # a specific profile is provided for use in CE
            af, M1_final = dewi_tauris2000(data, profile=profiles, **kwargs)

        else:
            ce_mn = data['model_number'][-1]
            profile_legend = profiles['legend']

            diff = abs(profile_legend['model_number'] - ce_mn)

            if np.min(diff) > max_profile_distance:
                # no suitable profile, use Webbink instead
                print('\t CE: Fallback to Webbink')
                al = kwargs.get('a_ce', 1)
                af, M1_final = webbink1984(data, al=al)

            else:
                profile_name = profile_legend['profile_name'][diff == np.min(diff)][0]
                profile = profiles[profile_name.decode('UTF-8')]

                af, M1_final = dewi_tauris2000(data, profile=profile, **kwargs)

    M2 = data['star_2_mass'][-1]

    G = 2944.643655  # Rsol^3/Msol/days^2
    P = np.sqrt(4 * np.pi ** 2 * af ** 3 / G * (M1_final + M2))

    q = M1_final / M2

    rl_1 = af * 0.49 * q ** (2.0 / 3.0) / (0.6 * q ** (2.0 / 3.0) + np.log(1 + q ** (1.0 / 3.0)))
    rl_2 = af * 0.49 * q ** (-2.0 / 3.0) / (0.6 * q ** (-2.0 / 3.0) + np.log(1 + q ** (-1.0 / 3.0)))

    # copy the last row of data so that we don't overwrite the parameters at the start of the CE
    # add a flag 'CE_phase' that is set to 1 during the CE phase. For now this is maximum 1 row.
    data = np.hstack([data, data[-1]])
    data['model_number'][-1] = data['model_number'][-1] + 1

    # update the parameters after the CE phase to their final version
    data['period_days'][-1] = P
    data['binary_separation'][-1] = af
    data['star_1_mass'][-1] = M1_final
    data['envelope_mass'][-1] = np.max([M1_final - data['he_core_mass'][-1], 0])  # can't be negative
    data['mass_ratio'][-1] = q
    data['rl_1'][-1] = rl_1
    data['rl_2'][-1] = rl_2
    data['CE_phase'][-1] = 1
    data['CE_phase'][-2] = 1

    return data


def iben_tutukov1984(history, al=1):
    """
    CE formalism from
    `Iben & Tutukov 1984, ApJ, 284, 719 <https://ui.adsabs.harvard.edu/abs/1984ApJ...284..719I/abstract>`_

    Required history parameters:
        - star_1_mass
        - star_2_mass
        - he_core_mass
        - binary_separation

    :param history: ndarray with model parameters
    :param al: alpha CE, the efficiency parameter for the CE formalism
    :return: final separation, final primary mass
    """
    M1 = history['star_1_mass'][-1]
    M2 = history['star_2_mass'][-1]
    Mc = history['he_core_mass'][-1]
    a = history['binary_separation'][-1]

    af = al * (Mc * M2) / (M1 ** 2) * a

    return af, Mc


def webbink1984(history, al=1, lb=1):
    """
    CE formalism from
    `Webbink 1984, ApJ, 277, 355 <https://ui.adsabs.harvard.edu/abs/1984ApJ...277..355W/abstract>`_

    Required history parameters:
        - star_1_mass
        - star_2_mass
        - he_core_mass
        - binary_separation
        - rl_1

    :param history: ndarray with model parameters
    :param al: alpha CE, the efficiency parameter for the CE formalism
    :param lb: lambda CE, the mass distribution factor of the primary envelope: lambda * Rl = the effective
               mass-weighted mean radius of the envelope at the start of CE.
    :return: final separation, final primary mass
    """
    M1 = history['star_1_mass'][-1] # Msun
    M2 = history['star_2_mass'][-1] # Msun
    Mc = history['he_core_mass'][-1] # Msun
    Me = M1 - Mc # Msun
    a = history['binary_separation'][-1] # Rsun
    Rl = history['rl_1'][-1]  # Rsun

    af = (a * al * lb * Rl * Mc * M2) / (2 * a * M1 * Me + al * lb * Rl * M1 * M2)

    return af, Mc


def demarco2011(history, al=1, lb=1):
    """
    CE formalism from
    `De Marco et al. 2011, MNRAS, 411, 2277 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.411.2277D/abstract>`_

    Required history parameters:
        - star_1_mass
        - star_2_mass
        - he_core_mass
        - binary_separation
        - rl_1

    :param history: ndarray with model parameters
    :param al: alpha CE, the efficiency parameter for the CE formalism
    :param lb: lambda CE, the mass distribution factor of the primary envelope: lambda * Rl = the effective
               mass-weighted mean radius of the envelope at the start of CE.
    :return: final separation, final primary mass
    """
    M1 = history['star_1_mass'][-1] # Msun
    M2 = history['star_2_mass'][-1] # Msun
    Mc = history['he_core_mass'][-1] # Msun
    Me = M1 - Mc # Msun
    a = history['binary_separation'][-1] # Rsun
    Rl = history['rl_1'][-1]  # Rsun

    af = (a * al * lb * Rl * Mc * M2) / (Me * (Me / 2.0 + Mc) * a + al * lb * Rl * M1 * M2)

    return af, Mc


def dewi_tauris2000(history, profile, a_ce=1, a_th=0.5, merge_when_core_reached=True):
    """
    CE formalism presented in
    `Dewi and Tauris 2000, A&A, 360, 1043 <https://ui.adsabs.harvard.edu/abs/2000A%26A...360.1043D/abstract>`_
    based on the idea of obtaining the binding energy by integrating the stellar profile from
    `Han et al 1995, MNRAS, 272, 800 <https://ui.adsabs.harvard.edu/abs/1995MNRAS.272..800H/abstract>`_


    Required history parameters:
        - star_2_mass
        - binary_separation

    Required profile parameters:
        - mass
        - logR
        - logP
        - logRho

    :param history: ndarray with model parameters
    :param profile: ndarray profile for the integration of binding energy
    :param a_ce: efficiency of ce
    :param a_th: efficiency of binding energy
    :param merge_when_core_reached: if True, the system is reported as a merger when the He core is reached in the
                                    iteration before the envelope is ejected and the CE ends.
    :return: final separation, final primary mass
    """

    def fRoche1(q):
        Xi = np.log10(q)
        ResPre = -0.420297 + 0.232069 * (Xi) - 0.0438153 * (Xi ** 2) - \
                 0.00567072 * (Xi ** 3) + 0.00870908 * (Xi ** 4) - 0.0205853 * (Xi ** 5) - \
                 0.0169055 * (Xi ** 6) + 0.0876934 * (Xi ** 7) - 0.0227041 * (Xi ** 8) - \
                 0.13918 * (Xi ** 9) + 0.118513 * (Xi ** 10) + 0.0627232 * (Xi ** 11) - \
                 0.122257 * (Xi ** 12) + 0.0345071 * (Xi ** 13) + 0.0297013 * (Xi ** 14) - \
                 0.0253245 * (Xi ** 15) + 0.00734239 * (Xi ** 16) - 0.000780009 * (Xi ** 17)
        return (pow(10, ResPre))

    M2 = history['star_2_mass'][-1]  # Msun
    Mc = history['he_core_mass'][-1]  # Msun
    a = history['binary_separation'][-1]  # Rsun
    G = 2944.643655  # Rsun^3/Msun/days^2

    star_outside_rl = True
    i = 0
    while star_outside_rl:
        line = profile[i]

        dm = line['mass'] - profile[i+1]['mass']
        M1 = profile[i+1]['mass']  # Msun
        R1 = 10**profile[i+1]['logR']  # Rsun
        Rmid = R1 + (10**line['logR'] - R1) / 2  # mid point of the cell in Rsol
        q = M1 / M2
        U = (3.0 * 10**line['logP']) / (2.0 * 10**line['logRho'])  # cm^2 / s^2
        U = U * 1.5432035916041713e-12  # Rsun^2 / days^2

        da = dm * (G * M1 / Rmid - a_th * U + a_ce * G * M2 / (2 * a)) * 2 * a**2 / (a_ce * G * M1 * M2)
        a = a - da

        i += 1

        # check if still outside RL
        #rl_1 = a * 0.49 * q**(2.0/3.0) / (0.6 * q**(2.0/3.0) + np.log(1 + q**(1.0/3.0)))
        rl_1 = a * fRoche1(q)
        if R1 < rl_1:
            star_outside_rl = False

        # if center of star reached report merger
        if i >= len(profile)-1:
            M1 = 0
            a = 0
            break

        # if core is reached and merge_when_core_reached == True report merger
        if merge_when_core_reached and M1 < Mc:
            M1 = Mc
            a = 0
            break

    M1_final = M1
    a_final = a

    return a_final, M1_final
