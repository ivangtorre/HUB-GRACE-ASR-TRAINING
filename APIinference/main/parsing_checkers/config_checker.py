import jsonschema

__author__ = "Ander González Docasal <agonzalezd@vicomtech.org>"


class ConfigError(Exception):
    pass


config_schema = {
    'type': 'object',
    'properties': {
        'AUTOPUNCT': {
            'type': 'object',
            'properties': {
                'model_path': {'type': 'string', "minLength": 1},
                'valid_seq_len': {'type': 'string'},
                'ctx_window_len': {'type': 'string'},
                'batch_size': {'type': 'string'},
            },
            'additionalProperties': False,
            'required': [
                'model_path'
            ],
        },
    },
    'required': [
        'AUTOPUNCT',
    ],
    'additionalProperties': False,
}

jsonschema.Draft4Validator.check_schema(config_schema)


def check_config(config):
    """
    :return: is_correct, error_message
    """
    validator = jsonschema.Draft4Validator(config_schema)
    error_messages = []
    for err in validator.iter_errors(config):
        path = '[' + ']['.join(err.path) + ']'
        error_messages.append('Error at parsing_checkers{}: {}'.format(path, err.message))

    if error_messages:
        raise ConfigError('\n'.join(error_messages))

    for param in ['valid_seq_len', 'ctx_window_len', 'batch_size']:
        try:
            config['AUTOPUNCT'][param] = int(config['AUTOPUNCT'][param])
        except ValueError:
            error_messages.append(f'"{param}" must be an integer value')

    if error_messages:
        raise ConfigError('\n'.join(error_messages))
