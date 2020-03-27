
import numpy as np

import pandas as pd

from numpy.lib.recfunctions import append_fields

from scipy.interpolate import interp1d

from nnaps.mesa import fileio, common_envelope, evolution_phases


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


def extract_parameters(data, parameters=[], phase_flags=[]):

    phases = []
    for parameter in parameters:
        pname, phase, func = evolution_phases.decompose_parameter(parameter)
        phases.append(phase)
    phases = phases + phase_flags
    phases = evolution_phases.get_all_phases(phases, data)

    result = []

    # extract the parameters
    for parameter in parameters:
        pname, phase, func = evolution_phases.decompose_parameter(parameter)

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


def extract_mesa(file_list, stability_criterion='J_div_Jdot_div_P', stability_limit=10,
                 ce_formalism='iben_tutukov1984', ce_parameters={'al':1}, ce_profile_name=None,
                 parameters=[], phase_flags=[], extra_info_parameters=[], verbose=False,**kwargs):

    parameters_, column_names = [], []
    for parameter in parameters:
        if type(parameter) == tuple:
            parameters_.append(parameter[0])
            column_names.append(parameter[1].strip())
        else:
            parameters_.append(parameter)
            column_names.append(parameter.strip())
    parameters = parameters_

    extra_parameters_, extra_names = [], []
    for parameter in extra_info_parameters:
        if type(parameter) == tuple:
            extra_parameters_.append(parameter[0])
            extra_names.append(parameter[1].strip())
        else:
            extra_parameters_.append(parameter)
            extra_names.append(parameter.strip())
    extra_info_parameters = extra_parameters_

    columns = ['path', 'stability'] + extra_names + column_names + phase_flags
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
        stable, ce_age = common_envelope.is_stable(data, criterion=stability_criterion, value=stability_limit)
        stability = 'stable'

        if not stable:
            # if the model is not stable, cut of the evolution at the start of the CE as anything after than
            # is non physical anyway.
            data = data[data['age'] <= ce_age]

            if ce_profile_name is not None:
                profiles = profiles[ce_profile_name]
            data = common_envelope.apply_ce(data, profiles=profiles, ce_formalism=ce_formalism, **ce_parameters)

            # check if CE is ejected or if the system is a merger
            if data['binary_separation'][-1] <= 0:
                stability = 'merger'
            else:
                stability = 'CE'

        # check if the binary is a contact system at any point during its evolution
        s = np.where((data['star_2_radius'] >= 0.99 * data['rl_2']) & (data['star_1_radius'] >= 0.99 * data['rl_1']))
        if len(data['model_number'][s]) > 0:
            stability = 'contact'

        # 3: extract some standard parameters
        pars = [model['path'].split('/')[-1]]
        pars += [stability]

        # 4: add the extra info to the output
        for p in extra_info_parameters:
            pars.append(extra_info[p])

        # 5: extract the requested parameters & 6: add the requested phase flags
        extracted_pars = extract_parameters(data, parameters, phase_flags)
        pars += extracted_pars

        # 7: todo: check for some possible errors and flag them

        results.append(pars)

    results = pd.DataFrame(results, columns=columns)
    return results



