
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


def count_ml_phases(data):
    """
    Count how many separate mass loss phases take place during the evolution of this system.
    A mass loss phase is defined as lg_mstar_dot_1 >= -10

    :param data: numpy ndarray containing the history of the system.
    :return: the number of mass loss phases
    """
    if all(data['lg_mstar_dot_1'] < -10):
        # no mass loss
        return 0

    ds = data.copy()

    ml_count = 0
    a = 0
    while a < ds['age'].max():
        try:
            a1 = ds['age'][ds['lg_mstar_dot_1'] >= -10][0]
        except IndexError:
            # no more new ML phases
            break
        ml_count += 1

        # select the first point in time that the mass loss dips below -10 after it  starts up.
        try:
            a = ds['age'][(ds['age'] > a1) & (ds['lg_mstar_dot_1'] < -10)][0]
        except IndexError:
            # No more new ML phases
            break

        ds = ds[ds['age'] > a]

    return ml_count


def check_error_flags(data, termination_code):
    """
    Check for some possible errors in the model and report them.

    Errors that are checked are:

        0: MESA stopped because of max model number
        1: MESA stopped because the accretor is overflowing it's Roche-lobe
        2: The mass loss phase ends before the evolution end. (To check if mass loss didn't cause mesa to crash).
        3: Potential problem with He ignition

    :param data: history data
    :param extracted_parameters: list of already extracted parameters
    :return: list of integers with error codes
    """

    error_codes = []

    # Check termination_code
    if termination_code == 'max_model_number':
        error_codes.append(0)
    if termination_code == 'accretor_overflow_terminate':
        error_codes.append(1)

    # Check if Mass loss ends before the end of the evolution
    if data['lg_mstar_dot_1'][-1] > -10:
        error_codes.append(2)

    # Check if there is a ramp-up to He ignition, but no actual ignition. Check the last 2000 points
    d = data[ data['age'] > data[data['log_LHe'] > -50]['age'][0] ]
    if len(d) > 5000:
        log_LHe = d['log_LHe'][-5000:]
        log_LHe_avg = np.average(log_LHe)
        if (log_LHe > log_LHe_avg-0.5).all() and (log_LHe < log_LHe_avg+0.5).all() \
                and log_LHe_avg > -50 and log_LHe_avg > data['log_LHe'][0]:
            error_codes.append(3)

    return error_codes


def process_file_list(file_list, verbose=False, **kwargs):
    """
    Function that takes a list containing the paths of the mesa models to process, and potentially some relevant
    parameters necessary for the processing. Check if the setup parameters provided in the kwargs are included in
    the list, and if not, add them.

    if ce_parameters is included in the file_list, it will convert the provided ce_parameters to dictionary.

    :param file_list: Pandas DataFrame containing at least 1 column with the path of the MESA models to process
    :param verbose: If True, print which parameters were added
    :param kwargs: all parameters that are required for processing the MESA models and their default values.
    :return: Pandas DataFrame containing the path to all models and the necessary parameters.
    """

    ce_parameters = kwargs.pop('ce_parameters', None)

    # add extraction parameters to the file_list
    for setup_par in kwargs.keys():
        if setup_par not in file_list.columns:
            file_list[setup_par] = kwargs[setup_par]
            if verbose:
                print('Set default parameter: {} = {} to all models'.format(setup_par, kwargs[setup_par]))

    # ce_parameters is treated separately because it should be converted from string to dictionary if already included
    if ce_parameters is not None and 'ce_parameters' not in file_list.columns:
        file_list['ce_parameters'] = [ce_parameters for i in file_list['path']]
        if verbose:
            print('Set default parameter: ce_parameters = {} to all models'.format(ce_parameters))
    else:
        file_list['ce_parameters'] = [eval(p) for p in file_list['ce_parameters']]

    return file_list


def extract_mesa(file_list, stability_criterion='J_div_Jdot_div_P', stability_limit=10,
                 ce_formalism='iben_tutukov1984', ce_parameters={'al':1}, ce_profile_name=None,
                 parameters=[], phase_flags=[], extra_info_parameters=[], add_setup_pars_to_result=True, verbose=False,
                 **kwargs):

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

    columns = ['path', 'stability', 'n_ML_phases'] + extra_names + column_names + phase_flags
    if add_setup_pars_to_result:
        columns += ['stability_criterion', 'stability_limit', 'ce_profile_name', 'ce_formalism', 'ce_parameters']
    columns += ['error_flags']
    results = []

    # check if the same extraction parameters are used for all models, or if specific parameters are already
    # provided in the files list
    file_list = process_file_list(file_list, stability_criterion=stability_criterion, stability_limit=stability_limit,
                 ce_formalism=ce_formalism, ce_parameters=ce_parameters, ce_profile_name=ce_profile_name,
                 verbose=verbose)

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
        stable, ce_age = common_envelope.is_stable(data, criterion=model['stability_criterion'],
                                                   value=model['stability_limit'])
        stability = 'stable'

        if not stable:
            # if the model is not stable, cut of the evolution at the start of the CE as anything after than
            # is non physical anyway.
            data = data[data['age'] <= ce_age]

            if ce_profile_name is not None:
                try:
                    profiles = profiles[model['ce_profile_name']]
                except Exception:
                    # todo: deal correctly with the missing profile!
                    print('CE: profile missing')
            data = common_envelope.apply_ce(data, profiles=profiles, ce_formalism=model['ce_formalism'],
                                            **model['ce_parameters'])

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

        # 3: extract some standard parameters: Path, stability and nr of ML phases.
        pars = [model['path'].split('/')[-1]]
        pars += [stability, count_ml_phases(data)]

        # 4: add the extra info to the output
        for p in extra_info_parameters:
            pars.append(extra_info[p])

        # 5: extract the requested parameters & 6: add the requested phase flags
        extracted_pars = extract_parameters(data, parameters, phase_flags)
        pars += extracted_pars

        # 7: Add the extraction setup parameters if requested
        if add_setup_pars_to_result:
            setup_pars = [model['stability_criterion'], model['stability_limit'], model['ce_profile_name'],
                          model['ce_formalism'], model['ce_parameters']]
            pars += setup_pars

        # 8: todo: check for some possible errors and flag them
        error_flags = check_error_flags(data, extra_info['termination_code'])
        pars.append(error_flags)

        results.append(pars)

    results = pd.DataFrame(results, columns=columns)
    return results



