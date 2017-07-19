from __future__ import print_function


class Registrar(object):
    
    """
    Register all parameters that can be used in pychrom
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

    # cadet solver parameters
    solver_parameters = set()
    solver_parameters.add('abstol')
    solver_parameters.add('algtol')
    solver_parameters.add('init_step_size')
    solver_parameters.add('max_steps')
    solver_parameters.add('reltol')
    solver_parameters.add('nthreads')

    solver_defaults = dict()
    solver_defaults['abstol'] = 1e-8
    solver_defaults['algtol'] = 1e-12
    solver_defaults['init_step_size'] = 1e-6
    solver_defaults['max_steps'] = 1e5
    solver_defaults['reltol'] = 1e-6
    solver_defaults['nthreads'] = 1


    #############################################
    # Section parameters
    section_parameters = dict()
    section_parameters['scalar'] = set()
    section_parameters['index'] = set()

    section_parameters['scalar'].add('start_time_sec')
    section_parameters['index'].add('const_coeff')
    section_parameters['index'].add('lin_coeff')
    section_parameters['index'].add('quad_coeff')
    section_parameters['index'].add('cube_coeff')

    section_parameters['scalar def'] = dict()
    section_parameters['index def'] = dict()
    for p in section_parameters['index']:
        section_parameters['index def'][p] = 0.0

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
    column_parameters['scalar'].add('binding')

    column_parameters['index'].add('init_c')
    column_parameters['index'].add('init_q')
    column_parameters['index'].add('film_diffusion')
    column_parameters['index'].add('par_diffusion')
    column_parameters['index'].add('par_surfdiffusion')
    #column_parameters['index'].add('cref')
    #column_parameters['index'].add('qref')

    column_parameters['scalar def'] = dict()

    column_parameters['index def'] = dict()
    column_parameters['index def']['init_c'] = 0.0
    column_parameters['index def']['init_q'] = 0.0
    #column_parameters['index def']['cref'] = 1.0
    #column_parameters['index def']['qref'] = 1.0
    column_parameters['index def']['par_surfdiffusion'] = 0.0

    for p in column_parameters['scalar']:
        scalar_parameters.add(p)

    for p in column_parameters['index']:
        single_index_parameters.add(p)

    for p, v in column_parameters['scalar def'].items():
        default_scalar_parameters[p] = v

    for p, v in column_parameters['index def'].items():
        default_single_index_parameters[p] = v

    ###############################################
    # discretization
    discretization_parameters = set()
    discretization_parameters.add('ncol')
    discretization_parameters.add('npar')
    discretization_parameters.add('nbound')
    discretization_parameters.add('par_disc_type')
    discretization_parameters.add('use_analytic_jacobian')
    #discretization_parameters.add('reconstruction')
    discretization_parameters.add('gs_type')
    discretization_parameters.add('max_krylov')
    discretization_parameters.add('max_restarts')
    discretization_parameters.add('schur_safety')

    discretization_defaults = dict()
    discretization_defaults['gs_type'] = 1
    discretization_defaults['max_krylov'] = 0
    discretization_defaults['max_restarts'] = 10
    discretization_defaults['schur_safety'] = 1e-8
    discretization_defaults['use_analytic_jacobian'] = 1
    discretization_defaults['par_disc_type'] = 'EQUIDISTANT_PAR'

    weno_parameters = set()
    weno_parameters.add('boundary_model')
    weno_parameters.add('weno_order')
    weno_parameters.add('weno_eps')

    weno_defaults = dict()
    weno_defaults['boundary_model'] = 0
    weno_defaults['weno_order'] = 3
    weno_defaults['weno_eps'] = 1e-8

    #############################################
    # ADSORPTION parameters

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
    adsorption_parameters['sma']['index'].add('sma_ka')
    adsorption_parameters['sma']['index'].add('sma_kd')
    adsorption_parameters['sma']['index'].add('sma_nu')
    adsorption_parameters['sma']['index'].add('sma_sigma')

    for p in adsorption_parameters['sma']['index']:
        single_index_parameters.add(p)

    for p, v in adsorption_parameters['sma']['index def'].items():
        default_scalar_parameters[p] = v

    ########## Multi-Component Langmuir

    adsorption_parameters['mcl'] = dict()
    adsorption_parameters['mcl']['scalar'] = set()
    adsorption_parameters['mcl']['index'] = set()

    adsorption_parameters['mcl']['scalar def'] = dict()
    adsorption_parameters['mcl']['index def'] = dict()

    # multi-component langmuir adsorption parameters
    # no scalar parameters for multi-component langmuir adsorption model

    # index parameters
    adsorption_parameters['mcl']['index'].add('mcl_ka')
    adsorption_parameters['mcl']['index'].add('mcl_kd')
    adsorption_parameters['mcl']['index'].add('mcl_qmax')

    for p in adsorption_parameters['mcl']['index']:
        single_index_parameters.add(p)

    for p, v in adsorption_parameters['mcl']['index def'].items():
        default_scalar_parameters[p] = v

    ########## Multi-Component Linear

    adsorption_parameters['lin'] = dict()
    adsorption_parameters['lin']['scalar'] = set()
    adsorption_parameters['lin']['index'] = set()

    adsorption_parameters['lin']['scalar def'] = dict()
    adsorption_parameters['lin']['index def'] = dict()

    # multi-component linear adsorption
    # no scalar parameters for multi-component linear adsorption model

    # index parameters
    adsorption_parameters['lin']['index'].add('lin_ka')
    adsorption_parameters['lin']['index'].add('lin_kd')

    for p in adsorption_parameters['lin']['index']:
        single_index_parameters.add(p)

    for p, v in adsorption_parameters['lin']['index def'].items():
        default_scalar_parameters[p] = v

