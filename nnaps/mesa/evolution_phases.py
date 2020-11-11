
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

AGREGATE_FUNCTIONS = ['max', 'min', 'avg', 'diff', 'rate']
EVOLUTION_PHASES = ['init', 'final', 'MS', 'RGB', 'ML', 'MLstart', 'MLend', 'CE', 'CEstart', 'CEend', 'HeIgnition',
                    'HeCoreBurning', 'HeShellBurning', 'sdA', 'sdB', 'sdO', 'He-WD']

#{ Load limits for core He burning

base_path = Path(__file__).parent
HeIgnition = pd.read_csv(base_path / 'helium_burn.data', sep='\s+', names=['rho', 'T'])
HeIgF = interpolate.interp1d(HeIgnition['rho'], HeIgnition['T'], bounds_error=False, fill_value='extrapolate')

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


def _check_history_parameters(history, parameters, evol_phase='UK', raise_error=True):
    """
    Check if all parameters are present in the evolution history of the model. Throws an error if this is not the case
    if raise_error is set to True. Otherwise the function will return True if all parameters are present, and False
    otherwise.
    Default behaviour is to raise a ValueError if a parameter is missing.

    :param history: the evolution history of the system
    :type history: numpy ndarray
    :param parameters: parameters that should be included
    :type parameters: list of strings
    :param evol_phase: the name of the phase that is doing the checking, used in the error message
    :type evol_phase: str
    :param raise_error: if true, raise an error, else return True if all parameters are present, False otherwise
    :type raise_error: bool
    :return: True if all parameters are present, False otherwise
    :rtype: bool
    """
    missing_parameters = []
    for par in parameters:
        if not par in history.dtype.names:
            missing_parameters.append(par)

    if len(missing_parameters) > 0 and raise_error:
        raise ValueError("""Evolution phase {} requires the following parameters {}, I could not find {} 
                            in the provided history file.""".format(evol_phase, parameters, missing_parameters))

    return len(missing_parameters) == 0


def _return_function(data, a1, a2, return_start=False, return_end=False, return_age=False):
    """
    Helper Function.
    Returns the entire phase, the start or the end depending on the keywords. Written as a separate function as this
    logic is the same for all phases.

    return_start and return_end can't both be True

    If both return_start and return_end are false, the entire phase is returned.

    return_age can be combined with both return_start, return_end or for the entire phase.
    """
    if return_start:
        if return_age:
            return a1
        else:
            s = np.where(data['age'] >= a1)
            return ([s[0][0]],)
    elif return_end:
        if return_age:
            return a2
        else:
            s = np.where(data['age'] <= a2)
            return ([s[0][-1]],)
    else:
        if return_age:
            return (a1, a2)
        else:
            return np.where((data['age'] >= a1) & (data['age'] <= a2))


def MS(data, return_age=False, return_start=False, return_end=False):
    """
    The Main sequence phase is defined as the phase where hydrogen burning takes place is the core.

    Specifically this is defined as the time period starting when the majority of the energy is produced by nuclear
    reactions, and ending when the core hydrogen runs out.

    start: 10^log_LH > 0.999 * 10^log_L

    end: center_h1 < 1e-12

    Required history parameters:
        - log_L
        - log_LH
        - center_h1
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the main sequence phase
    """
    required_parameters = ['log_L', 'log_LH', 'center_h1', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='MS')

    if not any(10**data['log_LH'] / 10**data['log_L'] > 0.999):
        # MS is not reached
        return None

    a1 = data['age'][(10**data['log_LH'] / 10**data['log_L'] > 0.999)][0]

    if not any(data['center_h1'] < 1e-12):
        a2 = data['age'][-1]
    else:
        a2 = data['age'][(data['center_h1'] < 1e-12)][0]

    return _return_function(data, a1, a2, return_age=return_age, return_start=return_start, return_end=return_end)

def MSstart(data, return_age=False):
    """
    The Main sequence phase is defined as the phase where hydrogen burning takes place is the core. MSstart defines
    first moment when the star enters the MS.

    Specifically this is defined as the moment when the majority of the energy is produced by nuclear reactions:

    start: log_LH >= 0.999 * log_L

    Required history parameters:
        - log_L
        - log_LH
        - center_h1
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the main sequence phase
    """
    return MS(data, return_age=return_age, return_start=True)


def MSend(data, return_age=False):
    """
    The Main sequence phase is defined as the phase where hydrogen burning takes place is the core. MSend defines
    last moment when the star is still considered to be on the MS.

    Specifically this is defined when the core hydrogen runs out:

    end: center_h1 <= 1e-12

    Required history parameters:
        - log_L
        - log_LH
        - center_h1
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the main sequence phase
    """
    return MS(data, return_age=return_age, return_end=True)


def RGB(data, return_age=False, return_start=False, return_end=False):
    """
    The red giant phase is defined as the phase starting at the end of the MS, and continuing until either a
    minimum in Teff or a maximum in luminosity is reached (whichever comes first) before He burning stars.

    Specifically, the start is defined in the same way as the end of the MS phase, based on central hydrogen, and
    the end is defined based on Teff and log_L before the central He fraction is reduced:

    start: center_h1 < 1e-12

    end: ( Teff == min(Teff) or log_L == max(log_L) ) and center_He >= center_He_TAMS - 0.01

    Required history parameters:
        - center_h1
        - center_he4
        - effective_T
        - log_L
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the red giant phase
    """
    required_parameters = ['log_L', 'center_h1', 'center_he4', 'effective_T', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='RGB')

    if not any(data['center_h1'] < 1e-12):
        # RGB phase never started
        return None

    a1 = data['age'][(data['center_h1'] < 1e-12)][0]

    # select data between start RGB and start of He burning.
    c_he_tams = data['center_he4'][(data['center_h1'] < 1e-12)][0]
    drgb = data[(data['center_he4'] >= c_he_tams - 0.01) & (data['center_h1'] < 1e-12)]

    a2 = drgb['age'][(drgb['effective_T'] == np.min(drgb['effective_T'])) | (drgb['log_L'] == np.max(drgb['log_L']))][0]

    return _return_function(data, a1, a2, return_age=return_age, return_start=return_start, return_end=return_end)


def RGBstart(data, return_age=False):
    """
    The start of the red giant phase is defined as the moment that the central hydrogen runs out. This phase is
    equivalent with the end of the MS phase:

    start: center_h1 <= 1e-12

    Required history parameters:
        - center_h1
        - center_he4
        - effective_T
        - log_L
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the red giant phase
    """
    return RGB(data, return_age=return_age, return_start=True)


def RGBend(data, return_age=False):
    """
    The end of the red giant phase is defined as the moment when either a minimum in Teff or a maximum in luminosity is
    reached (whichever comes first) before He burning stars.

    Specifically, the end is defined based on Teff and log_L before the central He fraction is reduced:

    end: ( Teff == min(Teff) or log_L == max(log_L) ) and center_He >= center_He_TAMS - 0.01

    Required history parameters:
        - center_h1
        - center_he4
        - effective_T
        - log_L
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the red giant phase
    """
    return RGB(data, return_age=return_age, return_end=True)


def HeIgnition(data, return_age=False):
    """
    The moment of He ignition, as defined by a peak in the He luminosity. This is the first moment of He ignition,
    but is not necessarily in the core as for low mass stars, He ignition occurs under degenerate conditions, and due
    to neutrino cooling typically happens in a shell around the core.

    Ignition is defined as the point with the maximum LHe between the first moment when LHe > 10 Lsol and the formation
    of the carbon-oxygen core. This is the (first) He flash.

    Required history parameters:
        - log_LHe
        - c_core_mass
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the moment of He ignition
    """
    required_parameters = ['log_LHe', 'c_core_mass', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='HeIgnition')

    if np.all(data['log_LHe'] < 1):
        # no He ignition
        return None
    a1 = data['age'][data['log_LHe'] > 1][0]

    if np.all(data['c_core_mass'] < 0.01):
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


def HeCoreBurning(data, return_age=False, return_start=False, return_end=False):
    """
    He core burning is defined as the period between ignition of He and formation of CO core.

    He ignition in the core is defined as the first moment when both the temperature and density in the core are
    sufficiently high to allow He burning. NNaPS uses the same conditions to define the ignition criteria as MESA.

    CO core formation is defined as the point in time when the CO core reaches as mass of 0.01

    If the center temperature and density are never in the range necessary for He fusion, there is no He core burning
    phase and the function will return None.

    Required history parameters:
        - log_center_T
        - log_center_Rho
        - log_LHe
        - c_core_mass
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the He core burning phase
    """
    required_parameters = ['log_center_T', 'log_center_Rho', 'log_LHe', 'c_core_mass', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='HeCoreBurning')

    if np.all(data['log_LHe'] < 1) or np.all(data['log_center_T'] < HeIgF(data['log_center_Rho'])):
        # no He ignition or no core burning
        return None
    a1 = data['age'][data['log_center_T'] >= HeIgF(data['log_center_Rho'])][0]

    if np.all(data['c_core_mass'] < 0.01):
        # model ignites He, but has problems modeling the core burning. No core burning phase can be returned
        return None

    a2 = data['age'][(data['age'] >= a1) & (data['c_core_mass'] <= 0.01)][-1]

    return _return_function(data, a1, a2, return_age=return_age, return_start=return_start, return_end=return_end)


def HeShellBurning(data, return_age=False, return_start=False, return_end=False):
    """
    The He shell burning phase is defined as the period in time between the formation of the CO core, and the final
    drop in He luminosity indicating the end of He burning. This final drop is defined as the time when LHe drops
    below half the LHe at the start of He shell burning.

    If there is no CO core present at any time, there is no He shell burning phase, and the function will return None

    start: CO core > 0.01 Msol

    end: LHe < 1/2 * LHe[start of shell burning]

    Required history parameters:
        - log_LHe
        - c_core_mass
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the He shell burning phase
    """
    required_parameters = [ 'log_LHe', 'c_core_mass', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='HeShellBurning')

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

    return _return_function(data, a1, a2, return_age=return_age, return_start=return_start, return_end=return_end)


def sdA(data):
    """
    This is the evolutionary definition of the sdA phase, which is defined as a core He burning phase where the star
    looks spectroscopically as an sdA star. This is defined as Teff between 15000 and 20000 K.

    If the star does not have a core He burning phase as defined by HeCoreBurning, this function will return None.

    .. note::
        Not to be confused with the spectroscopic definition of the sdA phase, which does NOT require He core burning.

    Required history parameters:
        - log_center_T
        - log_center_Rho
        - log_LHe
        - c_core_mass
        - log_Teff
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the sdA phase
    """
    required_parameters = ['log_center_T', 'log_center_Rho', 'log_LHe', 'c_core_mass', 'log_Teff', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='sdA')

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
    This is the evolutionary definition of the sdB phase, which is defined as a core He burning phase where the star
    looks spectroscopically as an sdB star. This is defined as Teff between 20000 and 40000 K.

    If the star does not have a core He burning phase as defined by HeCoreBurning, this function will return None.

    .. note::
        Not to be confused with the spectroscopic definition of the sdB phase, which does NOT require He core burning.

    Required history parameters:
        - log_center_T
        - log_center_Rho
        - log_LHe
        - c_core_mass
        - log_Teff
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the sdB phase
    """
    required_parameters = ['log_center_T', 'log_center_Rho', 'log_LHe', 'c_core_mass', 'log_Teff', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='sdB')

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
    This is the evolutionary definition of the sdB phase, which is defined as a core He burning phase where the star
    looks spectroscopically as an sdO star. This is defined as Teff higher than 40000 K.

    If the star does not have a core He burning phase as defined by HeCoreBurning, this function will return None.

    .. note::
        Not to be confused with the spectroscopic definition of the sdO phase, which does NOT require He core burning.

    Required history parameters:
        - log_center_T
        - log_center_Rho
        - log_LHe
        - c_core_mass
        - log_Teff
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the sdO phase
    """
    required_parameters = ['log_center_T', 'log_center_Rho', 'log_LHe', 'c_core_mass', 'log_Teff', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='sdO')

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
    Defines the He White Dwarf phase, when the star is on the WD cooling track, but still has a He core.

    The WD cooling track is selected to start when Teff < 10000K and logg > 7, or when logg > 7.5 regardless of Teff

    .. note::
        Is triggered with 'He-WD'.

    Required history parameters:
        - log_LHe
        - c_core_mass
        - log_Teff
        - log_g
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the He-WD phase
    """
    required_parameters = ['log_LHe', 'c_core_mass', 'log_Teff', 'log_g', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='He_WD')

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


def CO_WD(data):
    """
    Defines the He White Dwarf phase, when the star is on the WD cooling track, and has a CO core.

    The WD cooling track is selected to start when Teff < 10000K and logg > 7, or when logg > 7.5 regardless of Teff

    .. note::
        Is triggered with 'CO-WD'.

    Required history parameters:
        - log_LHe
        - c_core_mass
        - log_Teff
        - log_g
        - age

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the He-WD phase
    """
    required_parameters = ['log_LHe', 'c_core_mass', 'log_Teff', 'log_g', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='He_WD')

    if np.max(data['log_g']) < 7.0:
        # no final WD yet
        return None

    if np.max(data['c_core_mass']) < 0.01:
        # No He burning, so no CO core
        return None

    # select first point where teff < 10^4 and logg < 7
    a1 = data['age'][((data['log_Teff'] < 4) & (data['log_g'] > 7)) | (data['log_g'] >= 7.5)]
    if len(a1) == 0:
        # WD doesn't start
        return None
    else:
        a1 = a1[0]

    return np.where(data['age'] > a1)


def init(data):
    """
    First evolution time point, can be used to obtain the initial parameters of the run.

    :param data: numpy ndarray containing the history of the system.
    :return: selection of the first evolution point.
    """
    return ([0],)


def final(data):
    """
    Last evolution time point, can be used to obtain parameters at the very end of the run.

    :param data: numpy ndarray containing the history of the system.
    :return: selection of the last evolution point.
    """
    return ([data.shape[0]-1],)


def ML(data, mltype='rlof', return_start=False, return_end=False, return_age=False, return_multiple=False):
    """
    The first occurring mass loss phase, where the mass loss phase is defined as the period in time when the primary is
    losing mass at a rate of at least log(Mdot) >= -10.
    This phase only marks mass loss due to RLOF. Mass loss due to winds is not taken into account when flagging a ML
    phase. In practice, the mass loss rate due to RLOF is defined as:

        lg_mass_loss_rate = log10( 10^lg_mstar_dot_1 - 10^lg_wind_mdot_1 )

    .. note::
        This phase only marks the first occurring mass loss phase.

    Required history parameters:
        - lg_mstar_dot_1
        - lg_wind_mdot_1
        - age

    :param data: numpy ndarray containing the history of the system.
    :param mltype: the type of mass loss to consider: 'rlof', 'wind', 'total'
    :return: selection where the history corresponds to the first ML phase
    """
    required_parameters = ['lg_mstar_dot_1', 'lg_wind_mdot_1', 'age']
    _check_history_parameters(data, required_parameters, evol_phase='ML')

    # look only at the mass lost due to RLOF, not wind mass loss.
    if mltype == 'rlof':
        mass_loss = np.log10( 10**data['lg_mstar_dot_1'] - 10**data['lg_wind_mdot_1'] )
    elif mltype == 'wind':
        mass_loss = data['lg_wind_mdot_1'].copy()
    else:
        mass_loss = data['lg_mstar_dot_1'].copy()

    if all(mass_loss < -10):
        # no mass loss
        return None

    age = data['age'].copy()

    # run over history file, and measure all mass loss phases
    result_ages = []
    while True:
        try:
            a1 = age[mass_loss >= -10][0]
        except IndexError:
            # no more new ML phases
            break

        try:
            # select the first point in time that the mass loss dips below -10 after it
            # starts up. Necessary to deal with multiple mass loss phases.
            a2 = age[(age > a1) & (mass_loss < -10)][0]
        except IndexError:
            a2 = age[-1]

        result_ages.append((a1, a2))

        mass_loss = mass_loss[age > a2]
        age = age[age > a2]
        # stop loop if no more history remains, or if only 1 mass loss phase is requested.
        if len(age) == 0 or not return_multiple:
            break

    if not return_multiple:
        a1, a2 = result_ages[0]
        return _return_function(data, a1, a2, return_age=return_age, return_start=return_start, return_end=return_end)

    else:
        results = []
        for ages in result_ages:
            a1, a2 = ages
            results.append(_return_function(data, a1, a2, return_age=return_age,
                                            return_start=return_start, return_end=return_end))
        return results


def MLstart(data, mltype='rlof', return_age=False, return_multiple=False):
    """
    The start of the first ML phase, defined as the moment in time when the donor star first reaches
    lg_mstar_dot_1 >= 10

    Required history parameters:
        - lg_mstar_dot_1
        - age

    :param data: numpy ndarray containing the history of the system.
    :param mltype: the type of mass loss to consider: 'rlof', 'wind', 'total'
    :return: selection where the history corresponds to the starting point of the first ML phase
    """
    return ML(data, mltype=mltype, return_start=True, return_end=False,
              return_age=return_age, return_multiple=return_multiple)


def MLend(data, mltype='rlof', return_age=False, return_multiple=False):
    """
    The end of the first mass loss phase, defined as the moment in time when lg_mstar_dot_1 dips below -10 after
    starting mass loss.

    Required history parameters:
        - lg_mstar_dot_1
        - age

    :param data: numpy ndarray containing the history of the system.
    :param mltype: the type of mass loss to consider: 'rlof', 'wind', 'total'
    :return: selection where the history corresponds to the end point of the first ML phase
    """
    return ML(data, mltype=mltype, return_start=False, return_end=True,
              return_age=return_age, return_multiple=return_multiple)


def CE(data):
    """
    The CE phase as it is defined in the common_envelope settings.

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the CE phase
    """
    if all(data['CE_phase'] == 0):
        return None

    return np.where(data['CE_phase'] == 1)


def CEstart(data):
    """
    start of the CE phase as defined in the common_envelope settings.

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the starting point of the CE phase
    """
    s = CE(data)
    if s is None:
        return s

    return ([s[0][0]],)


def CEend(data):
    """
    the end of the CE phase as defined in the common_envelope settings.

    :param data: numpy ndarray containing the history of the system.
    :return: selection where the history corresponds to the end point of the CE phase
    """
    s = CE(data)
    if s is None:
        return s

    return ([s[0][-1]],)


all_phases = {'init': init, 'final': final,
              'MS': MS, 'MSstart': MSstart, 'MSend': MSend,
              'RGB': RGB, 'RGBstart': RGBstart, 'RGBend': RGBend,
              'MLstart': MLstart, 'MLend': MLend, 'ML': ML,
              'CEstart': CEstart, 'CEend': CEend, 'CE': CE,
              'HeIgnition': HeIgnition, 'HeCoreBurning': HeCoreBurning, 'HeShellBurning': HeShellBurning,
              'sdA': sdA, 'sdB': sdB, 'sdO': sdO, 'He-WD': He_WD, 'CO-WD': CO_WD}


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


def get_all_phases(phases, data, n_ml_phases=0):

    phases = set(phases)
    if None in phases:
        phases.remove(None)

    phase_selection = {}

    for phase in phases:
        if phase not in all_phases:
            phase_selection[phase] = get_custom_phase(phase, data)
        else:
            if 'ML' in phase and n_ml_phases > 0:
                # deal with multiple ML phases and store those as a list
                selection = all_phases[phase](data, return_multiple=True)
                phase_selection[phase] = selection[0:n_ml_phases]
            else:
                phase_selection[phase] = all_phases[phase](data)

    return phase_selection

#}