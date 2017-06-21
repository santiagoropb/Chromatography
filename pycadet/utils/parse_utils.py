from collections import OrderedDict
import pandas as pd
import numpy as np
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


def parse_scalar_inputs_from_dict(dict_inputs,
                                  class_name,
                                  registered_inputs,
                                  logger):

    parsed_scalar = dict()
    if dict_inputs is not None:
        for name, val in dict_inputs.items():
            msg = """{} is not a scalar parameter 
                        of {}""".format(name, class_name)
            assert name in registered_inputs, msg
            parsed_scalar[name] = val
    else:
        msg = """No scalar parameters specified 
                    when parsing {}""".format(class_name)
        logger.debug(msg)

    return parsed_scalar


def parse_index_components_from(obj,
                                dict_inputs,
                                map_name_to_id,
                                logger):

    has_params = False
    if dict_inputs is not None:

        if isinstance(dict_inputs, list):
            to_loop = sorted(dict_inputs)
        else:
            ordered_dict = OrderedDict(dict_inputs)
            to_loop = ordered_dict.keys()
            has_params = True

        for comp_name in to_loop:
            if comp_name in map_name_to_id.keys():
                msg = "Component {} being overwritten".format(comp_name)
                logger.warning(msg)
            obj.__add_component(comp_name)

    else:
        logger.warning("No components found to parsed")

    if len(obj._components) == 0:
        logger.warning("No components found to parsed")


def parse_parameters_indexed_by_components(inputs,
                                           map_name_to_id,
                                           registered_inputs,
                                           default_inputs):

    if isinstance(inputs, list):
        to_loop = sorted(inputs)
    elif isinstance(inputs, dict):
        ordered_dict = OrderedDict(inputs)
        to_loop = ordered_dict.keys()
    else:
        print(type(inputs))
        raise RuntimeError("Input not recognized")

    sublist_ids = []
    for cname in to_loop:
        if cname not in map_name_to_id.keys():
            msg = """ {} is not a component of Chromatography model
            """.format(cname)
            raise RuntimeError(msg)
        sublist_ids.append(map_name_to_id[cname])

    params_df = pd.DataFrame(index=sublist_ids,
                             columns=sorted(registered_inputs))

    # set defaults
    for name, default in default_inputs.items():
        params_df[name] = default

    params_df.index.name = 'component id'
    params_df.columns.name = 'parameters'

    for comp_name, params in inputs.items():
        comp_id = map_name_to_id[comp_name]
        for parameter, value in params.items():
            msg = """{} is not a valid parameter
                    of model""".format(parameter)
            assert parameter in registered_inputs, msg
            params_df.set_value(comp_id, parameter, value)

    return params_df
