import yaml

defaults_2h5 = {
    'star_columns': None,
    'binary_columns': None,
    'profile_columns': None,
    'add_stopping_condition': True,
    'input_path_kw': 'path',
    'input_path_prefix': '',
    'star1_history_file': 'LOGS/history1.data',
    'star2_history_file': 'LOGS/history2.data',
    'binary_history_file': 'LOGS/binary_history.data',
    'profile_files': None,
    'profiles_path': '',
    'profile_pattern': '*.profile',
    'log_file': 'log.txt',
    }

default_extract = {
    'stability_criterion': 'J_div_Jdot_div_P',
    'stability_limit': 10,
    'parameters': ['star_1_mass__init']
}


def read_defaults(filename):

    with open(filename, 'r') as stream:
        try:
            setup = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return setup