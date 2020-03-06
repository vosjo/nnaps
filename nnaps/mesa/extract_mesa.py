
import numpy as np

import pandas as pd

from numpy.lib.recfunctions import append_fields

from scipy.interpolate import interp1d

from nnaps.mesa import fileio, common_envelope


def read_history(objectname, return_profiles=False):

    data_ = fileio.read_hdf5(objectname)

    history = data_.pop('history', None)
    d1 = history.pop('star1', None)
    d2 = history.pop('star2', None)
    db = history.pop('binary', None)


    extra_info = data_.pop('extra_info', None)

    # set model number for primary to start at 1 and limits to correct last model number
    d1['model_number'] = d1['model_number'] - d1['model_number'][0] + 1
    s = np.where(db['model_number'] <= d1['model_number'][-1])
    db = db[s]

    # PRIMARY
    # now interpolate primary data to match model numbers for binary history
    dtypes = d1.dtype
    y = d1.view(np.float64).reshape(-1, len(dtypes))
    f = interp1d(d1['model_number'], y, axis=0, bounds_error=False, fill_value=0.0)
    d1 = f(db['model_number'])

    # reconvert from array to recarray
    d1 = [tuple(d) for d in d1]
    d1 = np.array(d1, dtype=dtypes)

    # remove model_number as column from d1 and merge into 1 recarray
    columns1 = list(d1.dtype.names)
    columns1.remove('model_number')
    column_names1 = [c for c in columns1]

    # SECONDARY
    # now interpolate secondary data to match model numbers for binary history
    dtypes = d2.dtype
    y = d2.view(np.float64).reshape(-1, len(dtypes))
    f = interp1d(d2['model_number'], y, axis=0, bounds_error=False, fill_value=0.0)
    d2 = f(db['model_number'])

    # reconvert from array to recarray
    d2 = [tuple(d) for d in d2]
    d2 = np.array(d2, dtype=dtypes)

    # remove model_number as column from d1 and merge into 1 recarray
    columns2 = list(d2.dtype.names)
    columns2.remove('model_number')
    column_names2 = [c+'_2' for c in columns2]

    # create a new record array from the data (much faster than appending to an existing array)
    columnsdb = list(db.dtype.names)
    data = np.core.records.fromarrays \
        ([db[c] for c in columnsdb] + [d1[c] for c in columns1] + [d2[c] for c in columns2],
                                      names=columnsdb + column_names1 + column_names2)

    if not 'effective_T' in data.dtype.names:
        data = append_fields(data, ['effective_T'], [10**data['log_Teff']], usemask=False)

    if not 'effective_T_2' in data.dtype.names:
        data = append_fields(data, ['effective_T_2'], [10**data['log_Teff_2']], usemask=False)

    if not 'rl_overflow_1' in data.dtype.names:
        data = append_fields(data, ['rl_overflow_1'], [data['star_1_radius'] / data['rl_1']], usemask=False)

    if not 'mass_ratio' in data.dtype.names:
        data = append_fields(data, ['mass_ratio'], [data['star_1_mass'] / data['star_2_mass']], usemask=False)

    if not 'separation_au' in data.dtype.names:
        data = append_fields(data, ['separation_au'], [data['binary_separation'] * 0.004649183820234682], usemask=False)

    J_Jdot_P = (data['J_orb'] / np.abs(data['Jdot'])) / (data['period_days'] * 24.0 *60.0 *60.0)
    J_Jdot_P = np.where((J_Jdot_P == 0 ), 99, np.log10(J_Jdot_P))
    data = append_fields(data, ['log10_J_div_Jdot_div_P'], [J_Jdot_P], usemask=False)

    M_Mdot_P = (data['star_1_mass'] / 10 ** data['lg_mstar_dot_1']) / (data['period_days'] / 360)
    M_Mdot_P = np.where((M_Mdot_P == 0), 99, np.log10(M_Mdot_P))
    data = append_fields(data, ['log10_M_div_Mdot_div_P'], [M_Mdot_P], usemask=False)

    if return_profiles:
        if 'profiles' not in data_:
            profiles = None
        else:
            profiles = data_.get('profiles')
            profiles['legend'] = data_.get('profile_legend')

        return data, extra_info, profiles

    return data, extra_info


def get_phases(data, phases):

    def get_phase(data, phase):
        if phase == 'init':
            return ([0],)

        elif phase == 'final':
            return ([data.shape[0]-1],)

        elif phase in ['MLstart', 'MLend', 'ML']:
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

            if phase == 'MLstart':
                s = np.where(data['age'] >= a1)
                return ([s[0][0]],)

            elif phase == 'MLend':
                s = np.where(data['age'] <= a2)
                return ([s[0][-1]],)

            else:
                return np.where((data['age'] >= a1) & (data['age'] <= a2))

        elif phase == 'HeIgnition':
            # select He ignition as the point between LHe > 10 Lsol and the formation of the
            # carbon-oxigen core where the L output is at it's maximum. This is the He flash.
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

            return np.where((data['log_LHe'] == np.max(d['log_LHe'])) & (data['age'] >= a1) & (data['age'] <= a2) )

        elif phase == 'HeCoreBurning':
            # He core burning is period between ignition of He and formation of CO core
            if np.all(data['log_LHe'] < 1):
                # no He ignition
                return None
            a1 = data['age'][data['log_LHe'] > 1][0]

            if np.all(data['c_core_mass'] < 0.01):
                # model ignites He, but has problems modeling the core burning. No core burning phase can be returned
                return None

            return np.where((data['age'] >= a1) & (data['c_core_mass'] <= 0.01))

        elif phase == 'HeShellBurning':
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
                    print (e)
                    a2 = data['age'][-1]

            return np.where((data['age'] >= a1) & (data['age'] <= a2))

        elif phase == 'sdB':
            # requires He core burning and 20000 < Teff < 40000
            if np.all(data['log_LHe'] < 1):
                # no He ignition
                return None
            a1 = data['age'][data['log_LHe'] > 1][0]

            if np.all(data['c_core_mass'] < 0.01):
                # model ignites He, but has problems modeling the core burning. No core burning phase can be returned
                return None

            return np.where((data['age'] >= a1) & (data['c_core_mass'] <= 0.01) &
                            (10**data['log_Teff'] >= 20000) & (10**data['log_Teff'] <= 40000))

        elif phase == 'sdO':
            # requires He burning and Teff > 40000
            if np.all(data['log_LHe'] < 1):
                # no He ignition
                return None
            a1 = data['age'][data['log_LHe'] > 1][0]

            if np.all(data['c_core_mass'] < 0.01):
                # model ignites He, but has problems modeling the core burning. No core burning phase can be returned
                return None

            return np.where((data['age'] >= a1) & (10 ** data['log_Teff'] >= 40000))

        elif phase == 'He-WD':
            # Requires star to be on WD cooling track and have He core

            if np.max(data['log_g']) < 7.5:
                # no final WD yet
                return None

            if np.max(data['c_core_mass']) > 0.01 or np.max(data['log_LHe']) > 1:
                # sign of He burning
                return None

            return np.where(data['log_g'] > 7.5)

    phase_selection = {}

    for phase in phases:

        phase_selection[phase] = get_phase(data, phase)

    return phase_selection


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
    def min(data, pname):
        return np.min(data[pname])

    def max(data, pname):
        return np.max(data[pname])

    def avg(data, pname):
        return np.average(data[pname], weights=10**data['log_dt'])

    def diff(data, pname):
        return data[pname][-1] - data[pname][0]

    def rate(data, pname):
        return (data[pname][-1] - data[pname][0]) / (data['age'][-1] - data['age'][0])

    known_functions = {'min': min, 'max':max, 'avg':avg, 'diff':diff, 'rate':rate}

    parts = par.split('__')

    pname, func, phase = None, None, None

    if len(parts) == 1:
        pname = parts[0]
        func = avg

    elif len(parts) == 2:
        pname = parts[0]
        if parts[-1] in known_functions.keys():
            func = known_functions[parts[1]]
        else:
            phase = parts[1]
            func = avg

    elif len(parts) == 3:
        pname = parts[0]
        phase = parts[1]
        func = known_functions[parts[2]]

    return pname, phase, func


def extract_parameters(data, parameters=[], phase_flags=[]):

    phases = ['init', 'final', 'MLstart', 'MLend', 'ML', 'HeIgnition', 'HeCoreBurning', 'sdB', 'sdO', 'He-WD']

    phases = get_phases(data, phases)

    result = []

    # extract the parameters
    for parameter in parameters:
        pname, phase, func = decompose_parameter(parameter)

        if phase is not None:
            if phases[phase] is None:
                # if the phase doesn't exist in the model, return nan value
                value = np.nan
            else:
                d_ = data[phases[phase]]
                value = func(d_, pname)
        else:
            d_ = data
            value = func(d_, pname)

        result.append(value)

    # add flags for triggered phases
    for phase in phase_flags:
        if phases[phase] is not None:
            result.append(True)
        else:
            result.append(False)

    return result


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


def apply_ce(data, ce_model=''):


    M1 = data['star_1_mass'][-1]
    M2 = data['star_2_mass'][-1]
    Mc = data['he_core_mass'][-1]
    a = data['binary_separation'][-1]

    af = 1.0 * a * (Mc * M2) / (M1 ** 2)
    G = 2944.643655  # Rsol^3/Msol/days^2
    P = np.sqrt(4 * np.pi ** 2 * af ** 3 / G * (Mc + M2))
    sma = af * 0.004649183820234682   # sma in AU
    sma_rsol = sma * 214.83390446073912
    q = Mc / M2

    rl_1 = sma_rsol * 0.49 * q**(2.0/3.0) / (0.6 * q**(2.0/3.0) + np.log(1 + q**(1.0/3.0)))
    rl_2 = sma_rsol * 0.49 * q**(-2.0/3.0) / (0.6 * q**(-2.0/3.0) + np.log(1 + q**(-1.0/3.0)))

    data['period_days'][-1] = P
    data['binary_separation'][-1] = sma
    data['star_1_mass'][-1] = Mc
    data['star_2_mass'][-1] = M2
    data['mass_ratio'][-1] = Mc/M2
    data['rl_1'][-1] = rl_1
    data['rl_2'][-1] = rl_2

    return data




def extract_mesa(file_list, stability_criterion='J_div_Jdot_div_P', stability_limit=10, parameters=[],
                 phase_flags=[], verbose=False):

    columns = ['path', 'stability'] + parameters + phase_flags
    # results = pd.DataFrame(columns=columns)
    results = []

    for i, model in file_list.iterrows():

        if verbose:
            print(i, model['path'])

        # 1: Get the data
        try:
            data, extra_info, profiles = read_history(model['path'], return_profiles=True)
        except Exception as e:
            if verbose:
                print(e)
            continue

        # 2: check for stability and cut data at start of CE
        stable, ce_age = is_stable(data, criterion=stability_criterion, value=stability_limit)
        if not stable:
            # if the model is not stable, cut of the evolution at the start of the CE and anything after than
            # is non physical anyway.
            data = data[data['age'] <= ce_age]

            data = common_envelope.apply_ce(data, ce_model='')

        # 3: extract some standard parameters
        pars = [model]
        pars += ['CE' if not stable else 'stable']  # todo: add contact binary and merger option here

        # 4: extract the requested parameters
        # 5: add the requested phase flags
        extracted_pars = extract_parameters(data, parameters, phase_flags)
        pars += extracted_pars

        # 6: check for some possible errors and flag them

        results.append(pars)

    results = pd.DataFrame(results, columns=columns)
    return results



