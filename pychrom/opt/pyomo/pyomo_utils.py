import pyomo.environ as pe

def CheckInstanceFeasibility(instance, tolerance):

    print("\n*** Feasibility check:")
    infeasibility_found = False

    for block in instance.block_data_objects():
        #for con in active_components_data(block, Constraint, sort_by_keys=True):
        for con in block.component_data_objects(pe.Constraint, active=True):
            #con.pprint()
            resid = ComputeConstraintResid(con)
            if (resid > tolerance):
                infeasibility_found = True
                PrintConstraintResid(con.name, con, resid)

    if infeasibility_found == False:
        print("   No infeasibilities found.")
    print("***\n")


def ComputeConstraintResid(con):
    bodyval = pe.value(con.body)
    upper_resid = 0
    if con.upper is not None:
        upper_resid = max(0, bodyval - pe.value(con.upper))
    lower_resid = 0
    if con.lower is not None:
        lower_resid = max(0, pe.value(con.lower) - bodyval)
    return  max(upper_resid, lower_resid)

def PrintConstraintResid(name, con, resid):
    if con.lower is None and con.upper is None:
        print('{0:10.4g} | {2:10s} <= {3:10.4g} <= {4:10s} : {1}'.format(resid, name, '-', pe.value(con.body), '-'))
    elif con.lower is None:
        print('{0:10.4g} | {2:10s} <= {3:10.4g} <= {4:10.4g} : {1}'.format(resid, name, '-', pe.value(con.body), pe.value(con.upper)))
    elif con.upper is None:
        print('{0:10.4g} | {2:10.4} <= {3:10.4g} <= {4:10s} : {1}'.format(resid, name, pe.value(con.lower), pe.value(con.body), '-'))
    else:
        print('{0:10.4g} | {2:10.4} <= {3:10.4g} <= {4:10.4g} : {1}'.format(resid, name, pe.value(con.lower), pe.value(con.body), pe.value(con.upper)))

def printStaleVariables(instance):
    print("\n*** Stale variables:")

    all_blocks_list = list(instance.block_data_objects())
    for block in all_blocks_list:
        variables = list(instance.component_data_objects(pe.Var))
        for var in variables:
            if var.stale == True:
                print(var)

def printVariablesHittingBounds(instance,tol=1-5):

    print("*** Variables that hit lb:")

    all_blocks_list = list(instance.block_data_objects())
    for block in all_blocks_list:
        variables = list(instance.component_data_objects(pe.Var))
        for var in variables:
            if var.lb is not None:
                if abs(var.value-var.lb)<=tol:
                    print(var, var.value, var.lb)
            if var.ub is not None:
                if abs(var.ub-var.value)<=tol:
                    print(var, var.value, var.ub)

    
    




