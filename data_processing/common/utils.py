
def to_sql_field(s):
    filter1 = s.replace(".","_").replace(" ","_")
    filter2 = ''.join(e for e in filter1 if e.isalnum())
    return filter2


def does_not_contain(token, value):
    """
    Validate that `token` is not a substring of `value`

    :param: token: string e.g. : | .
    :param: value: dictionary, list, or str
    """
    if isinstance(value, str):
        if token in value:
            raise ValueError("{value} cannot contain {token}.")

    if isinstance(value, list):
        if any([token in v for v in value]):
            raise ValueError(str(value) + " cannot contain {token}.")

    if isinstance(value, dict):
        if [token in k or token in v for k,v in value.items()]:
            raise ValueError(str(value) + " cannot contain {token}.")

    return True


def replace_token(token, token_replacement, value):
    """
    Replace `token` with `token_replacement` in `value`

    :param: token: string e.g. : | .
    :param: token_replacement: string e.g. _ -
    :param: value: dictionary, list, or str
    """
    if isinstance(value, str):
        return value.replace(token, token_replacement)

    if isinstance(value, list):
        new_value = []
        for v in value:
            new_value.append(v.replace(token, token_replacement))
        return new_value

    if isinstance(value, dict):
        new_value = {}
        for k,v in value.items():
            new_value[k.replace(token, token_replacement)] = v.replace(token, token_replacement)

        return new_value

    return value