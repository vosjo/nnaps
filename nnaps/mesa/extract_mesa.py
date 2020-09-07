
import numpy as np

import pandas as pd

from nnaps.mesa import fileio, common_envelope, evolution_phases

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
            data, extra_info, profiles = fileio.read_compressed_track(model['path'], return_profiles=True)
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

            # check if CE is ejected or if the system is a merger or a contact binary
            s = np.where((data['star_2_radius'] >= 0.99 * data['rl_2']) &
                         (data['star_1_radius'] >= 0.99 * data['rl_1']))

            if data['binary_separation'][-1] <= 0:
                stability = 'merger'
                print('CE: Merged')
            elif len(data['model_number'][s]) > 0:
                stability = 'contact'
                print('CE: Contact')
            else:
                stability = 'CE'

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



