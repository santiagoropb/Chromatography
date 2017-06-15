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
    multi_index_parameters = set()
    default_multi_index_parameters = dict()

    #############################################
    # scaling parameters
    single_index_parameters.add('qref')
    single_index_parameters.add('cref')
    default_single_index_parameters['qref'] = 1.0
    default_single_index_parameters['cref'] = 1.0

    #############################################
    # ADSORPTION parameters
    # TODO: in the future move adsorption parameters
    # to the multi_index set

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

    
    
