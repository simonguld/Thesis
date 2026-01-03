abp_data_dict = {
    'data_suffix': 'abp',
    'L_list': [128],
    'Nexp_list': [5],
    'act_exclude_dict': {128: []},
    'xlims': None,
    'uncertainty_multiplier': 5,
    'act_critical': None
}

abr_data_dict = {
    'data_suffix': 'abr',
    'L_list': [128],
    'Nexp_list': [5],
    'act_exclude_dict': {128: []},
    'xlims': None,
    'uncertainty_multiplier': 5,
    'act_critical': None
}

s_data_dict = {
'data_suffix': 's',
'L_list': [2048],
'Nexp_list': [3],
'act_exclude_dict': {2048: []},
'xlims': None,
'uncertainty_multiplier': 1,
'act_critical': 2.1
}

sd_data_dict = {
        'data_suffix': 'sd',
        'L_list': [512],
        'Nexp_list': [10],
        'act_exclude_dict': {512: []},
        'xlims': None,
        'uncertainty_multiplier': 20,
        'act_critical': 0.022
    }

na_data_dict = {
    'data_suffix': '',
    'L_list': [512, 1024, 2048],
    'Nexp_list': [5]*3,
    'act_exclude_dict': {512: [0.02, 0.0225, 0.0235], 1024: [], 2048: [0.0225]},
    'xlims': (0.016, 0.045),
    'uncertainty_multiplier': 20,
    'act_critical': 0.022
    }

na512_data_dict = {
    'data_suffix': '',
    'L_list': [512,],
    'Nexp_list': [5],
    'act_exclude_dict': {512: [0.02, 0.0225, 0.0235],},
    'xlims': (0.016, 0.045),
    'uncertainty_multiplier': 20,
    'act_critical': 0.022
    }
na1024_data_dict = {
    'data_suffix': '',
    'L_list': [1024,],
    'Nexp_list': [5],
    'act_exclude_dict': {1024: [],},
    'xlims': (0.016, 0.045),
    'uncertainty_multiplier': 20,
    'act_critical': 0.022
    }
na2048_data_dict = {
    'data_suffix': '',
    'L_list': [2048,],
    'Nexp_list': [5],
    'act_exclude_dict': {2048: [0.0225],},
    'xlims': (0.016, 0.045),
    'uncertainty_multiplier': 20,
    'act_critical': 0.022
    }

ndg_data_dict = {
    'data_suffix': 'ndg',
    'L_list': [1024],
    'Nexp_list': [1],
    'act_exclude_dict': {1024: []},
    'xlims': None,
    'uncertainty_multiplier': 20,
    'act_critical': 7
}

pol_data_dict = {
    'data_suffix': 'pol',
    'L_list': [2048],
    'Nexp_list': [1],
    'act_exclude_dict': {2048: [0.05, 0.1, 0.105,0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14,]},
    'xlims': None,
    'uncertainty_multiplier': 5,
    'act_critical': None
}

pols_data_dict = {
    'data_suffix': 'pols',
    'L_list': [2048],
    'Nexp_list': [1],
    'act_exclude_dict': {2048: []},
    'xlims': None,
    'uncertainty_multiplier': 1,
    'act_critical': None
}

DATA_CONFIGS = {'sd': sd_data_dict, 'ndg': ndg_data_dict, 'na': na_data_dict, 'na512':
                  na512_data_dict, 'na1024': na1024_data_dict, 'na2048': na2048_data_dict, 's': s_data_dict, 
                   'pol': pol_data_dict, 'pols': pols_data_dict, 'abp': abp_data_dict, 'abr': abr_data_dict}

FIG_FOLDER_CONFIGS = {
                'sd': 'sd',
                'ndg': 'ndg',
                'na': 'na',
                'na512': 'na512',
                'na1024': 'na1024', 
                'na2048': 'na2048',
                's': 's',
                'pol': 'pol',
                'pols': 'pols', 
                'abp': 'abp', 
                'abr': 'abr'}