import pyomo.environ as pe
import pyomo.dae as dae


def define_C_vars(model, scale_vars=True, with_second_der=False):

    if scale_vars:

        model.phi = pe.Var(model.s, model.t, model.x)
        model.dphidx = dae.DerivativeVar(model.phi, wrt=model.x)
        model.dphidt = dae.DerivativeVar(model.phi, wrt=model.t)

        def rule_rescale_c(m, i, j, k):
            return m.phi[i, j, k] * m.sc[i]

        def rule_rescale_dcdx(m, i, j, k):
            return m.dphidx[i, j, k] * m.sc[i]

        def rule_rescale_dcdt(m, i, j, k):
            return m.dphidt[i, j, k] * m.sc[i]

        # mobile phase concentration
        model.C = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_c)
        model.dCdx = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_dcdx)
        model.dCdt = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_dcdt)

        if with_second_der:
            model.dphidx2 = dae.DerivativeVar(model.phi, wrt=(model.x, model.x))

            def rule_rescale_dcdx2(m, i, j, k):
                return m.dphidx2[i, j, k] * m.sc[i]

            model.dCdx2 = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_dcdx2)
            
    else:

        model.C = pe.Var(model.s, model.t, model.x)
        model.dCdx = dae.DerivativeVar(model.C, wrt=model.x)
        model.dCdt = dae.DerivativeVar(model.C, wrt=model.t)
        if with_second_der:
            model.dCdx2 = dae.DerivativeVar(model.phi, wrt=(model.x, model.x))


def define_Cp_vars(model, scale_vars=True, index_radius=False, **kwargs):

    with_first_der = kwargs.pop('with_first_der', False)
    with_second_der = kwargs.pop('with_second_der', False)

    if not index_radius:
        if scale_vars:
            model.eta = pe.Var(model.s, model.t, model.x)
            model.dedt = dae.DerivativeVar(model.eta, wrt=model.t)

            def rule_rescale_cp(m, i, j, k):
                return m.eta[i, j, k] * m.sc[i]

            def rule_rescale_dcpdt(m, i, j, k):
                return m.dedt[i, j, k] * m.sc[i]

            # stationary phase concentration variable
            model.Cp = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_cp)
            model.dCpdt = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_dcpdt)
        else:
            model.Cp = pe.Var(model.s, model.t, model.x)
            model.dCpdt = dae.DerivativeVar(model.Cp, wrt=model.t)
    else:
        if scale_vars:
            model.eta = pe.Var(model.s, model.t, model.x, model.r)
            model.dedt = dae.DerivativeVar(model.eta, wrt=model.t)

            def rule_rescale_cp(m, i, j, k, w):
                return m.eta[i, j, k, w] * m.sc[i]

            def rule_rescale_dcpdt(m, i, j, k, w):
                return m.dedt[i, j, k, w] * m.sc[i]

            # stationary phase concentration variable
            model.Cp = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_cp)
            model.dCpdt = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_dcpdt)

            if with_first_der:
                model.dedr = dae.DerivativeVar(model.eta, wrt=model.r)

                def rule_rescale_dcpdr(m, i, j, k, w):
                    return m.dedr[i, j, k, w] * m.sc[i]

                model.dCpdr = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_dcpdr)

            if with_second_der:
                model.dedr2 = dae.DerivativeVar(model.eta, wrt=(model.r, model.r))

                def rule_rescale_dcpdr2(m, i, j, k, w):
                    return m.dedr2[i, j, k, w] * m.sc[i]

                model.dCpdr2 = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_dcpdr2)
        else:
            model.Cp = pe.Var(model.s, model.t, model.x, model.r)
            model.dCpdt = dae.DerivativeVar(model.Cp, wrt=model.t)

            if with_first_der:
                model.dCpdr = dae.DerivativeVar(model.Cp, wrt=model.r)
            if with_second_der:
                model.dCpdr2 = dae.DerivativeVar(model.Cp, wrt=(model.r, model.r))


def define_Q_vars(model, scale_vars=True, index_radius=False,  **kwargs):

    with_first_der = kwargs.pop('with_first_der', False)
    with_second_der = kwargs.pop('with_second_der', False)

    if not index_radius:
        if scale_vars:
            model.gamma = pe.Var(model.s, model.t, model.x)
            model.dgdt = dae.DerivativeVar(model.gamma, wrt=model.t)

            def rule_rescale_q(m, i, j, k):
                return m.gamma[i, j, k] * m.sq[i]

            def rule_rescale_dqdt(m, i, j, k):
                return m.dgdt[i, j, k] * m.sq[i]

            # stationary phase concentration variable
            model.Q = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_q)
            model.dQdt = pe.Expression(model.s, model.t, model.x, rule=rule_rescale_dqdt)
        else:
            model.Q = pe.Var(model.s, model.t, model.x)
            model.dQdt = dae.DerivativeVar(model.Q, wrt=model.t)
    else:
        if scale_vars:
            model.gamma = pe.Var(model.s, model.t, model.x, model.r)
            model.dgdt = dae.DerivativeVar(model.gamma, wrt=model.t)

            def rule_rescale_q(m, i, j, k, w):
                return m.gamma[i, j, k, w] * m.sq[i]

            def rule_rescale_dqdt(m, i, j, k, w):
                return m.dgdt[i, j, k, w] * m.sq[i]

            # stationary phase concentration variable
            model.Q = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_q)
            model.dQdt = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_dqdt)

            if with_first_der:
                model.dgdr = dae.DerivativeVar(model.gamma, wrt=model.r)

                def rule_rescale_dqdr(m, i, j, k, w):
                    return m.dgdr[i, j, k, w] * m.sq[i]

                model.dQdr = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_dqdr)

            if with_second_der:
                model.dgdr2 = dae.DerivativeVar(model.gamma, wrt=(model.r, model.r))

                def rule_rescale_dqdr2(m, i, j, k, w):
                    return m.dgdr2[i, j, k, w] * m.sq[i]

                model.dQdr2 = pe.Expression(model.s, model.t, model.x, model.r, rule=rule_rescale_dqdr2)
        else:
            model.Q = pe.Var(model.s, model.t, model.x, model.r)
            model.dQdt = dae.DerivativeVar(model.Q, wrt=model.t)

            if with_first_der:
                model.dQdr = dae.DerivativeVar(model.Q, wrt=model.r)
            if with_second_der:
                model.dQdr2 = dae.DerivativeVar(model.Q, wrt=(model.r, model.r))


def define_free_sites_vars(model, scale_vars=True, index_radius=False):

    if not index_radius:
        if scale_vars:
            model.theta = pe.Var(model.t, model.x)

            def rule_rescale_free_sites(m, j, k):
                return m.theta[j, k] * m.sf

            # stationary phase concentration variable
            model.free_sites = pe.Expression(model.t, model.x, rule=rule_rescale_free_sites)
        else:
            model.free_sites = pe.Var(model.t, model.x)
    else:
        if scale_vars:
            model.theta = pe.Var(model.t, model.x, model.r)

            def rule_rescale_free_sites(m, j, k, w):
                return m.theta[j, k, w] * m.sf

            # stationary phase concentration variable
            model.free_sites = pe.Expression(model.t, model.x, model.r, rule=rule_rescale_free_sites)
        else:
            model.free_sites = pe.Var(model.t, model.x, model.r)