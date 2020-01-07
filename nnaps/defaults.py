from sklearn import preprocessing

default_scaler = preprocessing.StandardScaler
default_encoder = preprocessing.OneHotEncoder

default_model = [{'layer':'Dense', 'args':[100], 'kwargs': dict(activation='relu', name='FC_1')},
                 {'layer':'Dropout', 'args':[0.1], 'kwargs': dict(name='DO_1')},
                 {'layer':'Dense', 'args':[50], 'kwargs': dict(activation='relu', name='FC_2')},
                 {'layer':'Dropout', 'args':[0.1], 'kwargs': dict(name='DO_2')},
                 {'layer':'Dense', 'args':[25], 'kwargs': dict(activation='relu', name='FC_3')},
                 {'layer':'Dropout', 'args':[0.1], 'kwargs': dict(name='DO_3')},
                 ]

def add_defaults_to_setup(setup):

    def convert_to_default_dict(setup_dict, key, default_dict):
        setup_dict = setup_dict.copy()

        def_key = list(default_dict.keys())[0]
        def_val = default_dict[def_key]

        if type(setup_dict[key]) is list:
            features = {}
            for f in setup_dict[key]:
                features[f] = {def_key: def_val}

            setup_dict[key] = features

        else:
            for f in setup_dict[key].keys():
                if def_key not in setup_dict[key][f]:
                    setup_dict[key][f][def_key] = def_val

        return setup_dict

    setup = convert_to_default_dict(setup, 'features', {'processor': default_scaler})
    setup = convert_to_default_dict(setup, 'regressors', {'processor': None})
    setup = convert_to_default_dict(setup, 'classifiers', {'processor': default_encoder})

    return setup