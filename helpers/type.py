def str_or_other(value):
    if value == 'None':
        return None
    elif value == 'True':
        return True
    elif value == 'False':
        return False
    return value