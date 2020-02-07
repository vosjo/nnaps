
import numpy as np

import pandas as pd

from numpy.lib.recfunctions import append_fields

from scipy.interpolate import interp1d
from . import fileio

def read_history(objectname):

    data_ = fileio.read_hdf5(objectname)

    history = data_.pop('history', None)
    d1 = history.pop('star1', None)
    d2 = history.pop('star2', None)
    db = history.pop('binary', None)

    # zinit = data_['z']
    # fehinit = data_['feh']
    # population = data_['population']
    # pinit_frac = data_['pinit_frac']
    # gal_age = data_['gal_age']
    # termination_code = data_['termination_code']

    # set model number for primary to start at 1 and limits to correct last model number
    d1['model_number'] = d1['model_number'] - d1['model_number'][0] + 1
    s = np.where(db['model_number'] <= d1['model_number'][-1])
    db = db[s]

    # PRIMARY
    # now interpolate primary data to match model numbers for binary history
    dtypes = d1.dtype
    y = d1.view(np.float64).reshape(d1.shape + (-1,))
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
    y = d2.view(np.float64).reshape(d2.shape + (-1,))
    f = interp1d(d2['model_number'], y, axis=0, bounds_error=False, fill_value=0.0)
    d2 = f(db['model_number'])

    # reconvert from array to recarray
    d2 = [tuple(d) for d in d2]
    d2 = np.array(d2, dtype=dtypes)

    # remove model_number as column from d1 and merge into 1 recarray
    columns2 = list(d2.dtype.names)
    columns2.remove('model_number')
    column_names2 = [ c +'_2' for c in columns2]

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

    if not 'q' in data.dtype.names:
        data = append_fields(data, ['q'], [data['star_1_mass'] / data['star_2_mass']], usemask=False)

    if not 'separation_au' in data.dtype.names:
        data = append_fields(data, ['separation_au'], [data['binary_separation'] * 0.004649183820234682], usemask=False)

    J_Jdot_P = (data['J_orb'] / np.abs(data['Jdot'])) / (data['period_days'] * 24.0 *60.0 *60.0)
    J_Jdot_P = np.where((J_Jdot_P == 0 ), 99, np.log10(J_Jdot_P))
    data = append_fields(data, ['log10_J_div_Jdot_div_P'], [J_Jdot_P], usemask=False)


    return data #, zinit, fehinit, population, pinit_frac, gal_age, termination_code

def get_phases(data, phases):

    def get_phase(data, phase):
        if phase == 'init':
            return ([0],)

        elif phase == 'final':
            return ([data.shape[0]-1],)

        elif phase == 'MLstart':
            s = np.where(data['lg_mstar_dot_1'] > -10)

            return ([s[0][0]],)

        elif phase == 'MLend':
            a = data['age'][data['lg_mstar_dot_1'] > -10][0]
            s = np.where((data['age'] > a) & (data['lg_mstar_dot_1'] <= -10))

            return ([s[0][0]],)

        elif phase == 'ML':
            a1 = data['age'][data['lg_mstar_dot_1'] > -10][0]
            a2 = data['age'][(data['age'] > a1) & (data['lg_mstar_dot_1'] <= -10)][0]

            s = np.where((data['age'] >= a1) & (data['age'] <= a2))
            return s

        elif phase == 'HeIgnition':
            # select He ignition as the point between LHe > 10 Lsol and the formation of the
            # carbon-oxigen core where the L output is at it's maximum. This is the He flash.
            if np.all(data['log_LHe'] < 1):
                # no He ignition
                return None
            a1 = data['age'][data['log_LHe'] > 1][0]
            a2 = data['age'][data['c_core_mass'] >= 0.01][0]
            d = data[(data['age'] >= a1) & (data['age'] <= a2)]

            return np.where(d['log_LHe'] == np.max(d['log_LHe']))

        elif phase == 'HeCoreBurning':
            # He core burning is period between ignition of He and formation of CO core
            if np.all(data['log_LHe'] < 1):
                # no He ignition
                return None
            a1 = data['age'][data['log_LHe'] > 1][0]

            return np.where((data['age'] > a1) & (data['c_core_mass'] < 0.01))

        # elif phase == 'HeShellBurning':
        #     # He shell burning is period between formation of CO core and the drop in He luminocity
        #     a1 = data['age'][data['c_core_mass'] >= 0.01][0]
        #
        #     sel = ([len(data['star_1_mass'])],)

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

def extract_parameters(data, parameters):

    phases = ['init', 'final', 'MLstart', 'MLend', 'ML', 'HeIgnition', 'HeCoreBurning']

    phases = get_phases(data, phases)

    result = {}

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

        result[parameter] = value

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


def get_post_ce_parameters(data, ce_model=''):

    M1 = data['star_1_mass'][-1]
    M2 = data['star_2_mass'][-1]
    Mc = data['he_core_mass'][-1]
    a = data['binary_separation'][-1]

    af = 1.0 * a * (Mc * M2) / (M1 ** 2)
    G = 2944.643655  # Rsol^3/Msol/days^2
    P = np.sqrt(4 * np.pi ** 2 * af ** 3 / G * (Mc + M2))
    sma = af * 0.004649183820234682 * 214.83390446073912  # sma in Rsol
    q = Mc / M2

    pars_ce = {
        'period_days__CE': P,
        'binary_separation__CE': sma,
        'star_1_mass__CE': Mc,
        'star_2_mass__CE': M2,
        'mass_ratio__CE': q,
    }

    return pars_ce


def extract_mesa(modellist, stability_criterion='J_div_Jdot_div_P', stability_limit=10, parameters=None):

    for i, model in modellist.iterrows():
        # 1: Get the data
        try:
            data = read_history(model)
        except Exception as e:
            print(e)
            continue

        # 2: check for stability and cut data at start of CE
        is_stable, ce_age = is_stable(data, criterion=stability_criterion, value=stability_limit)
        if not is_stable:
            # if the model is not stable, cut of the evolution at the start of the CE and anything after than
            # is non physical anyway.
            data = data[data['age'] <= ce_age]

            ce_pars = get_post_ce_parameters(data, ce_model='')

        # 3: extract some standard parameters
        pars = {}
        pars['stability'] = 'CE' if not is_stable else 'stable'  # todo: add contact binary and merger option here

        # 4: extract the requested parameters
        for parameter in parameters:
            pars_ = extract_parameters(data, parameters)




