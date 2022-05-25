import jsonschema

__author__ = "Ander González Docasal <agonzalezd@vicomtech.org>"


class JSONError(Exception):
    pass


request_schema = {
    'type': 'object',
    'properties': {
        'content': {'type': 'string', 'minLength': 1},
        'add_caps': {'type': 'boolean'}
    },
    'required': [
        'content',
    ],
    'additionalProperties': False,
}

jsonschema.Draft4Validator.check_schema(request_schema)


words_schema = {
    'type': 'object',
    'properties': {
        'content': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'start': {'type': 'number'},
                    'duration': {'type': 'number'},
                    'word': {'type': 'string'},
                    'conf': {'type': 'number'},
                },
                'additionalProperties': False,
            }
        },
        'add_caps': {'type': 'boolean'}
    },
    'required': [
        'content',
    ],
    'additionalProperties': False,
}

jsonschema.Draft4Validator.check_schema(words_schema)


def check_schema(request_json, json_schema):
    validator = jsonschema.Draft4Validator(json_schema)
    error_messages = []
    for err in validator.iter_errors(request_json):
        path = '[' + ']['.join(err.path) + ']'
        error_messages.append('Error at json{}: {}'.format(path, err.message))

    if error_messages:
        raise JSONError('\n'.join(error_messages))


def check_request(request_json):
    check_schema(request_json, request_schema)


def check_words(request_json):
    check_schema(request_json, words_schema)
