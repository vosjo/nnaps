
defaults_2h5 = {
    'star_columns': None,
    'binary_columns': None,
    'add_stopping_condition': True,
    'input_path_kw': 'path',
    'input_path_prefix': '',
    'star1_history_file': 'LOGS/history1.data',
    'star2_history_file': 'LOGS/history2.data',
    'binary_history_file': 'LOGS/binary_history.data',
    'log_file': 'log.txt',
    }

default_extract = {
    'stability_criterion': 'J_div_Jdot_div_P',
    'stability_limit': 10,
    'parameters': ['star_1_mass__init']
}