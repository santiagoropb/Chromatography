import yaml
import six

def parse_inputs(inputs):
    """
    Parse inputs
    :param inputs: filename or dictionary with inputs to the binding model
    :return: dictionary with parsed inputs
    """
    if isinstance(inputs, dict):
        args = inputs
    elif isinstance(inputs, six.string_types):
        if ".yml" in inputs or ".yaml" in inputs:
            with open(inputs, 'r') as f:
                args = yaml.load(f)
        else:
            raise RuntimeError('File format not implemented yet. Try .yml or .yaml')
    else:
        raise RuntimeError('inputs must be a dictionary or a file')

    return args
