
import numpy as np

def is_stable(data, criterion='J_div_Jdot_div_P', value=10):
    if criterion == 'Mdot':

        if np.max(data['lg_mstar_dot_1']) > value:
            s = np.where(data['lg_mstar_dot_1'] > value)
            a = data['age'][s][1]
            return False, a

    elif criterion == 'delta':

        if np.max(data['mass_transfer_delta']) > value:
            s = np.where(data['mass_transfer_delta'] > value)
            a = data['age'][s][1]
            return False, a

    elif criterion == 'J_div_Jdot_div_P':

        if np.min(10 ** data['log10_J_div_Jdot_div_P']) <= value:
            s = np.where(10 ** data['log10_J_div_Jdot_div_P'] < value)
            a = data['age'][s][1]
            return False, a

    elif criterion == 'M_div_Mdot_div_P':

        if np.min(10 ** data['log10_M_div_Mdot_div_P']) <= value:
            s = np.where(10 ** data['log10_M_div_Mdot_div_P'] < value)
            a = data['age'][s][1]
            return False, a

    elif criterion == 'R_div_SMA':

        if np.max(data['star_1_radius'] / data['binary_separation']) > value:
            s = np.where(data['star_1_radius'] / data['binary_separation'] > value)
            a = data['age'][s][1]
            return False, a

    return True, data['age'][-1]


def apply_ce(data, profile=None, ce_model='iben_tutukov1984', **kwargs):
    """
    Function performs the ce ejection and updates some stellar and binary parameters

    requires: star_1_mass, star_2_mass, he_core_mass, binary_separation
    updates: star_1_mass, star_2_mass, period_days, binary_separation, mass_ratio, rl_1, rl_2

    :param data: ndarray with model parameters
    :param ce_model: CE model to use (not implemented yet)
    :return: same dataset as provided with on the last line the parameters after the CE phase.
    """

    if ce_model == 'iben_tutukov1984':
        af, M1_final = iben_tutukov1984(data, **kwargs)

    elif ce_model == 'webbink1984':
        af, M1_final = webbink1984(data, **kwargs)

    elif ce_model == 'profile':
        af, M1_final = apply_ce_profile(data, profile=profile, **kwargs)

    else:
        af, M1_final = iben_tutukov1984(data, al=1)

    M2 = data['star_2_mass'][-1]

    G = 2944.643655  # Rsol^3/Msol/days^2
    P = np.sqrt(4 * np.pi ** 2 * af ** 3 / G * (M1_final + M2))
    sma = af * 0.004649183820234682  # sma in AU
    sma_rsol = sma * 214.83390446073912

    q = M1_final / M2

    rl_1 = sma_rsol * 0.49 * q ** (2.0 / 3.0) / (0.6 * q ** (2.0 / 3.0) + np.log(1 + q ** (1.0 / 3.0)))
    rl_2 = sma_rsol * 0.49 * q ** (-2.0 / 3.0) / (0.6 * q ** (-2.0 / 3.0) + np.log(1 + q ** (-1.0 / 3.0)))

    data['period_days'][-1] = P
    data['binary_separation'][-1] = sma
    data['star_1_mass'][-1] = M1_final
    data['mass_ratio'][-1] = q
    data['rl_1'][-1] = rl_1
    data['rl_2'][-1] = rl_2

    return data


def iben_tutukov1984(data, al=1):
    """
    CE formalism from Iben & Tutukov 1984, ApJ, 284, 719
    https://ui.adsabs.harvard.edu/abs/1984ApJ...284..719I/abstract

    :param data: ndarray with model parameters
    :param al: alpha CE, the efficiency parameter for the CE formalism
    :return: final separation, final primary mass
    """
    M1 = data['star_1_mass'][-1]
    M2 = data['star_2_mass'][-1]
    Mc = data['he_core_mass'][-1]
    a = data['binary_separation'][-1]

    print('A', a)
    print('M1', M1)
    print('M2', M2)
    print('Mc', Mc)

    af = al * (Mc * M2) / (M1 ** 2) * a

    return af, Mc


def webbink1984(data, al=1, lb=1):
    """
    CE formalism from Webbink 1984, ApJ, 277, 355
    https://ui.adsabs.harvard.edu/abs/1984ApJ...277..355W/abstract

    :param data: ndarray with model parameters
    :param al: alpha CE, the efficiency parameter for the CE formalism
    :param lb: lambda CE, the mass distribution factor of the primary envelope: lambda * Rl = the effective
               mass-weighted mean radius of the envelope at the start of CE.
    :return: final separation, final primary mass
    """
    M1 = data['star_1_mass'][-1] # Msun
    M2 = data['star_2_mass'][-1] # Msun
    Mc = data['he_core_mass'][-1] # Msun
    Me = M1 - Mc # Msun
    a = data['binary_separation'][-1] # Rsun
    Rl = data['rl_1'][-1]  # Rsun

    af = (a * al * lb * Rl * Mc * M2) / (2 * a * M1 * Me + al * lb * Rl * M1 * M2)

    return af, Mc


def demarco2011(data, al=1, lb=1):
    """
    CE formalism from De Marco et al. 2011, MNRAS, 411, 2277
    https://ui.adsabs.harvard.edu/abs/2011MNRAS.411.2277D/abstract

    :param data: ndarray with model parameters
    :param al: alpha CE, the efficiency parameter for the CE formalism
    :param lb: lambda CE, the mass distribution factor of the primary envelope: lambda * Rl = the effective
               mass-weighted mean radius of the envelope at the start of CE.
    :return: final separation, final primary mass
    """
    M1 = data['star_1_mass'][-1] # Msun
    M2 = data['star_2_mass'][-1] # Msun
    Mc = data['he_core_mass'][-1] # Msun
    Me = M1 - Mc # Msun
    a = data['binary_separation'][-1] # Rsun
    Rl = data['rl_1'][-1]  # Rsun

    af = (a * al * lb * Rl * Mc * M2) / (Me * (Me / 2.0 + Mc) * a + al * lb * Rl * M1 * M2)

    return af, Mc


def apply_ce_profile(data, profile, al=1, lb=1):

    # import pylab as pl
    # pl.plot(10**profile['logR'], profile['mass'])
    # s = np.where(profile['mass'] <= 0.378)
    # pl.axvline(x=10**profile['logR'][s][0])
    # pl.show()

    M2 = data['star_2_mass'][-1]
    a = data['binary_separation'][-1]

    alpha_th = 0
    alpha_ce = 1

    star_outside_rl = True
    i = 0
    while star_outside_rl:
        line = profile[i]

        dm = line['mass'] - profile[i+1]['mass']
        M1 = line['mass'] # Msol
        G = 2944.643655  # Rsol^3/Msol/days^2
        R1 = 10**line['logR'] # Rsol
        q = M1 / M2
        U = 0

        da = dm * (G * M1 / R1 + alpha_ce * G * M2 / (2 * a)) * 2 * a**2 / (alpha_ce * G * M1 * M2)
        a = a - da

        i += 1

        # check if still outside RL
        rl_1 = a * 0.49 * q**(2.0/3.0) / (0.6 * q**(2.0/3.0) + np.log(1 + q**(1.0/3.0)))
        if R1 < rl_1:
            star_outside_rl = False

    M1_final = M1
    a_final = a

    return a_final, M1_final