from __future__ import print_function


class Registrar(object):
    
    """
    Register all parameters that can be used in pycadet
    """
    
    # container with scalar parameters
    scalar_parameters = set()
    default_scalar_parameters = dict()
    # container with single index parameters
    single_index_parameters = set()
    default_single_index_parameters = dict()
    # container with multi index parameters
    # TODO: current design does not consider multi-index
    # TODO: this is left as an extension
    multi_index_parameters = set()
    default_multi_index_parameters = dict()

    # TODO: add dictionary with description of parameters

    #############################################
    # scaling parameters
    default_single_index_parameters['qref'] = 1.0
    default_single_index_parameters['cref'] = 1.0

    for k in default_single_index_parameters.keys():
        single_index_parameters.add(k)

    #############################################
    # Column parameters

    column_parameters = dict()
    column_parameters['scalar'] = set()
    column_parameters['index'] = set()

    column_parameters['scalar'].add('col_length')
    column_parameters['scalar'].add('col_porosity')
    column_parameters['scalar'].add('par_porosity')
    column_parameters['scalar'].add('par_radius')
    column_parameters['scalar'].add('col_dispersion')
    column_parameters['scalar'].add('velocity')

    column_parameters['index'].add('init_c')
    column_parameters['index'].add('init_pc')
    column_parameters['index'].add('init_q')
    column_parameters['index'].add('film_diffusion')
    column_parameters['index'].add('par_diffusion')

    column_parameters['scalar def'] = dict()

    column_parameters['index def'] = dict()
    column_parameters['index def']['init_c'] = 0.0
    column_parameters['index def']['init_q'] = 0.0
    column_parameters['index def']['init_cp'] = 0.0

    for p in column_parameters['scalar']:
        scalar_parameters.add(p)

    for p in column_parameters['index']:
        single_index_parameters.add(p)

    for p, v in column_parameters['scalar def'].items():
        default_scalar_parameters[p] = v

    for p, v in column_parameters['index def'].items():
        default_single_index_parameters[p] = v
        

    #############################################
    # ADSORPTION parameters
    # TODO: in the future move adsorption parameters
    # TODO: to the multi_index set

    ########## SMA
    adsorption_parameters = dict()
    adsorption_parameters['sma'] = dict()
    adsorption_parameters['sma']['scalar'] = set()
    adsorption_parameters['sma']['index'] = set()

    adsorption_parameters['sma']['scalar def'] = dict()
    adsorption_parameters['sma']['index def'] = dict()

    # SMA parameters
    # scalar
    adsorption_parameters['sma']['scalar'].add('sma_lambda')
    adsorption_parameters['sma']['scalar'].add('sma_cref')
    adsorption_parameters['sma']['scalar'].add('sma_qref')
    adsorption_parameters['sma']['scalar def']['sma_cref'] = 1.0
    adsorption_parameters['sma']['scalar def']['sma_qref'] = 1.0

    for p in adsorption_parameters['sma']['scalar']:
        scalar_parameters.add(p)

    for p, v in adsorption_parameters['sma']['scalar def'].items():
        default_scalar_parameters[p] = v

    # index parameters
    adsorption_parameters['sma']['index'].add('sma_kads')
    adsorption_parameters['sma']['index'].add('sma_kdes')
    adsorption_parameters['sma']['index'].add('sma_nu')
    adsorption_parameters['sma']['index'].add('sma_sigma')

    for p in adsorption_parameters['sma']['index']:
        single_index_parameters.add(p)

    for p, v in adsorption_parameters['sma']['index def'].items():
        default_scalar_parameters[p] = v

