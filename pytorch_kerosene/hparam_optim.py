from itertools import product


def params2str(params):
    output = '-'.join([f'{k}_{v}' for k, v in params.items()])
    output = output.replace('.', '').replace(',', '')
    return output


def generate_search_params(params_values):
    search_params = {}

    for param, value in params_values.items():
        if not isinstance(value, list) and not isinstance(value, tuple):
            params_values[param] = [value]

    for combination in product(*tuple(params_values.values())):
        params_dict = dict(zip(params_values.keys(), combination))
        search_params[params2str(params_dict)] = params_dict
    
    return search_params