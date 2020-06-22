
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sc

#{ Load limits for core He burning

base_path = Path(__file__).parent
HeIgnition = pd.read_csv(base_path / 'helium_burn.data', sep='\s+', names=['rho', 'T'])
HeIgF = sc.interpolate.interp1d(HeIgnition['rho'], HeIgnition['T'])

#}

#{ Parameter decomposition

def min_(data, pname):
    return np.min(data[pname])


def max_(data, pname):
    return np.max(data[pname])


def avg_(data, pname):
    return np.average(data[pname], weights=10 ** data['log_dt'])


def diff_(data, pname):
    return data[pname][-1] - data[pname][0]


def rate_(data, pname):
    return (data[pname][-1] - data[pname][0]) / (data['age'][-1] - data['age'][0])


known_functions = {'min': min_, 'max': max_, 'avg': avg_, 'diff': diff_, 'rate': rate_}


def decompose_parameter(par):
    """
    Decompose a parameter in the parameter name recognized by mesa, a potential function
    to apply to the parameter and the phase for which to calculate it.

    recognized functions:
     - min
     - max
     - avg (average over time)
     - diff (difference between start and end)
     - rate (change rate: (end-start)/(end_time - start_time)


    examples:
    M1__init                -> avg(star_1_mass[init])
    max__RL                 -> max(RL)
    duration__HeCoreBurning -> duration(time[HeCoreBurning])
    max__Teff__ML            -> max(effective_T[ML])
    """
    parts = par.split('__')

    pname, func, phase = None, None, None

    if len(parts) == 1:
        pname = parts[0]
        func = avg_

    elif len(parts) == 2:
        pname = parts[0]
        if parts[-1] in known_functions.keys():
            func = known_functions[parts[1]]
        else:
            phase = parts[1]
            func = avg_

    elif len(parts) == 3:
        pname = parts[0]
        phase = parts[1]
        func = known_functions[parts[2]]

    return pname, phase, func

#}

#{ Evolution Phases

def init(data):
    """
    First evolution time point
    """
    return ([0],)


def final(data):
    """
    Last evolution time point
    """
    return ([data.shape[0]-1],)

def ML(data, return_age=False):
    """
    Mass loss is defind as lg_mstar_dot_1 >= 10
    """
    if all(data['lg_mstar_dot_1'] < -10):
        # no mass loss
        return None

    a1 = data['age'][data['lg_mstar_dot_1'] >= -10][0]

    try:
        # select the first point in time that the mass loss dips below -10 after it
        # starts up. Necessary to deal with multiple mass loss phases.
        a2 = data['age'][(data['age'] > a1) & (data['lg_mstar_dot_1'] < -10)][0]
    except IndexError:
        a2 = data['age'][-1]

    if return_age:
        return a1, a2
    else:
        return np.where((data['age'] >= a1) & (data['age'] <= a2))


def MLstart(data, return_age=False):
    """
    First time lg_mstar_dot_1 reaches -10
    """
    ages = ML(data, return_age=True)
    if ages is None:
        return None
    else:
        a1, a2 = ages

    s = np.where(data['age'] >= a1)

    if return_age:
        return a1
    else:
        return ([s[0][0]],)


def MLend(data, return_age=False):
    """
    First time lg_mstar_dot_1 dips below -10 after starting mass loss
    """
    ages = ML(data, return_age=True)
    if ages is None:
        return None
    else:
        a1, a2 = ages

    s = np.where(data['age'] <= a2)

    if return_age:
        return a2
    else:
        return ([s[0][-1]],)


def CE(data):
    """
    CE phase as defined in the common_envelope settings.
    """
    if all(data['CE_phase'] == 0):
        return None

    return np.where(data['CE_phase'] == 1)


def CEstart(data):
    """
    start of the CE phase as defined in the common_envelope settings.
    """
    s = CE(data)
    if s is None:
        return s

    return ([s[0][0]],)


def CEend(data):
    """
    end of the CE phase as defined in the common_envelope settings.
    """
    s = CE(data)
    if s is None:
        return s

    return ([s[0][-1]],)


def HeIgnition(data, return_age=False):
    """
    select He ignition as the point between LHe > 10 Lsol and the formation of the
    carbon-oxigen core where the L output is at it's maximum. This is the He flash.
    """
    if np.all(data['log_LHe'] < 1):
        # no He ignition
        return None
    a1 = data['age'][data['log_LHe'] > 1][0]

    if np.all(data['c_core_mass']) < 0.01:
        # model ignites He, but has problems modeling the core burning. He ignition can be returned.
        a2 = data['age'][-1]
    else:
        a2 = data['age'][data['c_core_mass'] >= 0.01][0]

    d = data[(data['age'] >= a1) & (data['age'] <= a2)]
    s = np.where((data['log_LHe'] == np.max(d['log_LHe'])) & (data['age'] >= a1) & (data['age'] <= a2))

    if return_age:
        return data['age'][s][0]
    else:
        return s


def HeCoreBurning(data, return_age=False):
    """
    He core burning is period between ignition of He and formation of CO core
    """
    if np.all(data['log_LHe'] < 1) or np.all(data['log_center_T'] < HeIgF(data['log_center_Rho'])):
        # no He ignition or no core burning
        return None
    a1 = data['age'][data['log_center_T'] >= HeIgF(data['log_center_Rho'])][0]

    if np.all(data['c_core_mass'] < 0.01):
        # model ignites He, but has problems modeling the core burning. No core burning phase can be returned
        return None

    if return_age:
        a2 = data['age'][(data['age'] >= a1) & (data['c_core_mass'] <= 0.01)][-1]
        return a1, a2
    else:
        return np.where((data['age'] >= a1) & (data['c_core_mass'] <= 0.01))


def HeShellBurning(data, return_age=False):
    """
    Shell burning is taken as the period between the formation of the CO core and the drop in He luminosity
    """
    if np.all(data['log_LHe'] < 1):
        # no He ignition
        return None

    if np.all(data['c_core_mass'] < 0.01):
        # no actual core He burning takes place, so no shell burning either.
        return None

    a1 = data['age'][data['c_core_mass'] >= 0.01][0]
    LHe_burning = data['log_LHe'][data['age'] == a1][0]

    if len(data['age'][(data['age'] > a1) & (data['log_LHe'] < LHe_burning / 2.)]) > 0:
        a2 = data['age'][(data['age'] > a1) & (data['log_LHe'] < LHe_burning / 2.)][0]
    else:
        try:
            # end of He shell burning when carbon core gets almost its final mass
            a2 = data['age'][data['c_core_mass'] >= 0.98 * np.max(data['c_core_mass'])][0]
        except Exception as e:
            print(e)
            a2 = data['age'][-1]

    if return_age:
        return a1, a2
    else:
        return np.where((data['age'] >= a1) & (data['age'] <= a2))


def sdA(data):
    """
    The sdA phase requires core He burning and the average He core burning Teff between 15000 and 20000

    If the star is an sdA, returns the part of the He core burning phase with 15000 <= teff < 20000
    """
    ages = HeCoreBurning(data, return_age=True)

    # Core He Burning phase is required
    if ages is None:
        return None
    else:
        a1, a2 = ages

    d = data[(data['age'] > a1) & (data['age'] < a2)]

    teff = 10**avg_(d, 'log_Teff')

    if teff < 15000 or teff >= 20000:
        return None
    else:
        return np.where((data['age'] > a1) & (data['age'] < a2) &
                    (10 ** data['log_Teff'] >= 15000) & (10 ** data['log_Teff'] < 20000))


def sdB(data):
    """
    The sdB phase requires core He burning and the average He core burning Teff between 20000 and 40000

    If the star is an sdB, returns the part of the He core burning phase with 20000 <= teff < 40000
    """
    ages = HeCoreBurning(data, return_age=True)

    # Core He Burning phase is required
    if ages is None:
        return None
    else:
        a1, a2 = ages

    d = data[(data['age'] > a1) & (data['age'] < a2)]

    teff = 10 ** avg_(d, 'log_Teff')

    if teff < 20000 or teff >= 40000:
        return None
    else:
        return np.where((data['age'] > a1) & (data['age'] < a2) &
                    (10**data['log_Teff'] >= 20000) & (10**data['log_Teff'] < 40000))


def sdO(data):
    """
    sdO requires core He burning and the average He core burning Teff >= 40000

    If the star is an sdO, returns the part of the He core burning phase with teff >= 40000
    """
    ages = HeCoreBurning(data, return_age=True)

    # Core He Burning phase is required
    if ages is None:
        return None
    else:
        a1, a2 = ages

    d = data[(data['age'] > a1) & (data['age'] < a2)]

    teff = 10 ** avg_(d, 'log_Teff')

    if teff < 40000:
        return None
    else:
        return np.where((data['age'] > a1) & (data['age'] < a2) & (10**data['log_Teff'] >= 40000))


def He_WD(data):
    """
    Requires star to be on WD cooling track and have He core. Cooling track is selected to start when
    teff < 10000K and logg > 7, or when logg > 7.5 regardless of teff
    """

    if np.max(data['log_g']) < 7.0:
        # no final WD yet
        return None

    if np.max(data['c_core_mass']) > 0.01 or np.max(data['log_LHe']) > 1:
        # sign of He burning
        return None

    # select first point where teff < 10^4 and logg < 7
    a1 = data['age'][((data['log_Teff'] < 4) & (data['log_g'] > 7)) | (data['log_g'] >= 7.5)]
    if len(a1) == 0:
        # WD doesn't start
        return None
    else:
        a1 = a1[0]

    return np.where(data['age'] > a1)


all_phases = {'init': init, 'final': final, 'MLstart': MLstart, 'MLend': MLend, 'ML': ML,
              'CEstart': CEstart, 'CEend': CEend, 'CE': CE,
              'HeIgnition': HeIgnition, 'HeCoreBurning': HeCoreBurning, 'HeShellBurning': HeShellBurning,
              'sdA': sdA, 'sdB': sdB, 'sdO': sdO, 'He-WD': He_WD}


def get_custom_phase(phase, data):
    """
    Select phases that are linked to other parameters as for example maximum mass loss:
    lg_mstar_dot_1_max will return the point in time where the mass loss rate of the primary reaches it's maximum value.

    :param phase: phase (string)
    :param data: evolution data (np ndarray)
    :return: np selection
    """
    par = '_'.join(phase.split('_')[0:-1])
    func = phase.split('_')[-1]

    # check if both the parameter and the function are known
    if par not in data.dtype.names or func not in known_functions:
        return None

    func = known_functions[func]

    value = func(data, par)

    return np.where(data[par] == value)


def get_all_phases(phases, data):

    phases = set(phases)
    if None in phases:
        phases.remove(None)

    phase_selection = {}

    for phase in phases:
        if phase not in all_phases:
            phase_selection[phase] = get_custom_phase(phase, data)
        else:
            phase_selection[phase] = all_phases[phase](data)

    return phase_selection

#}