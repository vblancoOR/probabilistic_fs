from gurobipy import *
import numpy as np
import pandas as pd
import statistics as stats

def generate_nodes(tree_depth):
    nodes = list(range(1, int(round(2 ** (tree_depth + 1)))))
    parent_nodes = nodes[0: 2 ** (tree_depth + 1) - 2 ** tree_depth - 1]
    leaf_nodes = nodes[-2 ** tree_depth:]
    return parent_nodes, leaf_nodes


def get_parent(i,D):
    assert i > 1, "No parent for Root"
    assert i <= 2 ** (D + 1), "Error! Total: {0}; i: {1}".format(
        2 ** (D + 1), i)
    return int(i / 2)


def get_ancestors(i,D):
    assert i > 1, "No ancestors for Root"
    assert i <= 2 ** (D + 1), "Error! Total: {0}; i: {1}".format(
        2 ** (D + 1), i)
    left_ancestors = []
    right_ancestors = []
    j = i
    while j > 1:
        if j % 2 == 0:
            left_ancestors.append(int(j / 2))
        else:
            right_ancestors.append(int(j / 2))
        j = int(j / 2)
    return left_ancestors, right_ancestors

def pfs(train, C, p, p_plus, trees_vars, trees_vals, tree_depth,
    leaf_classes, timelimit, desired_class=1, epsilon=0.001):

    # trees_vars: each column contains the variable used in node t in tree r.
    # trees_vals: each column contains the threshold used in node t in tree r.
    #  dim = (n_parent_nodes,n_trees)

    ################ Model

    m = Model("PFS max-path")

    ################ Parameters

    K = range(2)
    preds = train.shape[1] - 1
    P = range(preds)


    n_trees = trees_vars.shape[0]
    n_vartree = trees_vars.shape[1]

    D = range(tree_depth)
    R = range(n_trees)
    parent_nodes, leaf_nodes = generate_nodes(tree_depth)


    # number of times that variable involved in node t of tree r
    # appears in the forest
    b = [0]*preds
    for j in range(n_vartree):
        for r in R:
            b[trees_vars.iloc[r,j]] += 1

    x = train.iloc[:,0:preds].values # x_train
    yy = train.iloc[:,-1].values # y_train

    ################ Model variables

    pfs_x = {}
    for j in P:
        pfs_x[j] = m.addVar(lb=0,vtype=GRB.CONTINUOUS, name='exp'+str(j))


    z = {}
    phi = {}
    delta = {}

    for r in R:
        delta[r] = m.addVar(vtype=GRB.BINARY, name='delta' + str(r) )

        for t in leaf_nodes:
            z[r,t] = m.addVar(vtype=GRB.BINARY, name='z' + str(r) +
                                                       '_' + str(t))

            phi[r,t] = m.addVar(vtype=GRB.BINARY,name='phi' + str(r) +
                                                       '_' + str(t))
    beta = {}
    for r in R:
        for j in P:
            beta[r,j] = m.addVar(vtype=GRB.BINARY, name='z' + str(r) +
                                                      '_' + str(j))

    m.update()

    ################ Model auxiliar variables

    omega = {}
    big_alpha = {}
    alpha = {}
    gamma = {}
    theta = {}
    sum_var = {}
    log_sumvar = {}
    for r in R:
        sum_var[r]= m.addVar(vtype=GRB.CONTINUOUS,name='sum_var' +
                                str(r))
        log_sumvar[r]= m.addVar(lb=-GRB.INFINITY,vtype=GRB.CONTINUOUS,name='logsum_var' +
                                        str(r))
        for t in parent_nodes:
            for l in leaf_nodes:
                omega[r,t,l] = m.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,name='omega' +
                                    str(r) + '_' + str(t))
        for t in leaf_nodes:
            big_alpha[r,t]= m.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,
                                    name='big_aplha' + str(r) + '_' + str(t))
            theta[r,t]= m.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,
                                    name='theta' + str(r) + '_' + str(t))
            for j in D:
                alpha[r,t,j] =  m.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,
                                        name='aplha'+ str(r) + '_' + str(t)
                                        + '_' + str(j))
                gamma[r,t,j] =  m.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,
                                        name='gamma'+ str(r) + '_' + str(t)
                                        + '_' + str(j))


    m.update()

    ################ Objective Function

    obj = quicksum(log_sumvar[r] for r in R)

    m.setObjective(obj, GRB.MAXIMIZE)
    m.setParam("TimeLimit", timelimit)
    # to be set up as required
    m.Params.NumericFocus = 1
    m.Params.OptimalityTol = 0.000001
    m.Params.IntFeasTol = 0.000001
    m.update()


    ################ Model Constraints

    # fix to 0 betas not used
    for j in P:
        if b[j]==0:
            m.addConstr(beta[r,j]==0, name="C_aux1[%d]"%(j+1))
            m.addConstr(pfs_x[j]==train.iloc[0,j],name="C_aux2[%d]"%(j+1))


    ### C 1
    m.addConstr(quicksum(quicksum(beta[r,trees_vars.iloc[r,t-1]]/b[trees_vars.iloc[r,t-1]] for
                t in parent_nodes) for r in R) <= C, name="C_01")

    ### C 2
    for r in R:
        for rr in R:
            for t in parent_nodes:
                for tt in parent_nodes:
                    if trees_vars.iloc[r,t-1]==trees_vars.iloc[rr,tt-1]:
                        m.addConstr(beta[r,trees_vars.iloc[r,t-1]] ==
                                    beta[rr,trees_vars.iloc[rr,tt-1]],
                                        name="C02[%d,%d,%d,%d]"%(r+1,rr+1,
                                                                t+1,tt+1))

    ### C 3
    # to be set up as required
    bigM=10
    for r in R:
        for l in leaf_nodes:
            left_ancestors, right_ancestors = get_ancestors(l,tree_depth)
            for t in left_ancestors:
                    m.addConstr(pfs_x[trees_vars.iloc[r,t-1]] - bigM*(1-z[r,l])+
                    epsilon <= trees_vals.iloc[r,t-1], name="C03a[%d,%d,%d]"%(r+1,
                                                l+1,t+1))
            for tt in right_ancestors:
                    m.addConstr(pfs_x[trees_vars.iloc[r,tt-1]] + bigM*(1-z[r,l]) -
                    epsilon >= trees_vals.iloc[r,tt-1], name="C03b[%d,%d,%d]"%(r+1,
                                                l+1,tt+1))

    ### C 4
    for r in R:
        m.addConstr(quicksum(z[r,l] for l in leaf_nodes) == 1,
                                        name="C04[%d]"%(r+1))

    ### C 5
    for k in K:
        if k != desired_class:
            m.addConstr(quicksum(quicksum(z[r,l] for l in
            leaf_nodes if leaf_classes.iloc[r,l-leaf_nodes[0]]==desired_class) for r in R) >=
            quicksum(quicksum(z[r,l]  for l in
            leaf_nodes if leaf_classes.iloc[r,l-leaf_nodes[0]]==k) for r in R), name="C05[%d]"%(k+1))

    ### C 6
    m.addConstr(quicksum(delta[r] for r in R) <=
                 -(n_trees/2 +1) + quicksum(quicksum(z[r,l] for l in leaf_nodes if
                                               leaf_classes.iloc[r,l-leaf_nodes[0]]== desired_class)
                                      for r in R), name="C06")

    ### C 7
    for r in R:
        m.addConstr(delta[r]  <=
                    quicksum(z[r,l] for l in leaf_nodes if
                    leaf_classes.iloc[r,l-leaf_nodes[0]] == desired_class),
                    name="C07[%d]"%(r+1))

    ### C 8
    for r in R:
        for l in leaf_nodes:
            m.addConstr(phi[r,l]<=desired_class - leaf_classes.iloc[r,l-leaf_nodes[0]],
                        name="C08[%d,%d]"%(k+1,l+1))

    # Note: desired_class takes the greatest value in the target variable

    m.update()

    ################ Aux Constraints
    for r in R:
        for t in parent_nodes:
            for l in leaf_nodes:
                left_ancestors, right_ancestors = get_ancestors(l,tree_depth)
                if t in right_ancestors:
                    m.addConstr(omega[r,t,l]<= p.iloc[r,t-1]*(1-beta[r,trees_vars.iloc[r,t-1]]) +
                                     p_plus.iloc[r,t-1]*beta[r,trees_vars.iloc[r,t-1]],
                                     name="Caux3[%d,%d]"%(r+1,t+1))
                else:
                    m.addConstr(omega[r,t,l]<= 1-p.iloc[r,t-1]*(1-beta[r,trees_vars.iloc[r,t-1]]) +
                                     (1-p_plus.iloc[r,t-1])*beta[r,trees_vars.iloc[r,t-1]],
                                     name="Caux4[%d,%d]"%(r+1,t+1))


    for r in R:
        for t in leaf_nodes:
            left_ancestors, right_ancestors = get_ancestors(t,tree_depth)
            lista = left_ancestors+right_ancestors
            lista.sort()

            m.addConstr(big_alpha[r,t]==alpha[r,t,0],
                            name="Caux5a[%d,%d]"%(r+1,t+1))
            for d in D:
                index = lista[d]
                if index in right_ancestors:
                    pp = p.iloc[r,index-1]
                    pp_plus = p_plus.iloc[r,index-1]

                if index in left_ancestors:
                    pp = 1-p.iloc[r,index-1]
                    pp_plus = 1-p_plus.iloc[r,index-1]

                if d == (tree_depth-1):
                    m.addConstr(alpha[r,t,d]==omega[r,get_parent(t,tree_depth),t],
                                        name="Caux5b[%d,d%d,%d]"%(r+1,t+1,d+1))

                else:
                    m.addConstr(alpha[r,t,d]<=pp*alpha[r,t,d+1]+
                                gamma[r,t,d]*(pp_plus-pp),
                                        name="Caux5c[%d,d%d,%d]"%(r+1,t+1,d+1))
                    m.addConstr(gamma[r,t,d]<=beta[r, trees_vars.iloc[r,index-1]],
                                        name="Caux5d[%d,d%d,%d]"%(r+1,t+1,d+1))
                    m.addConstr(gamma[r,t,d]<=alpha[r,t,d+1] ,
                                        name="Caux5e[%d,d%d,%d]"%(r+1,t+1,d+1))

    for r in R:
        for t in leaf_nodes:
            m.addConstr(theta[r,t]<=z[r,t],
                                name="Caux6a[d%d,%d]"%(r+1,t+1))
            m.addConstr(theta[r,t]<=big_alpha[r,t] + phi[r,t] + delta[r],
                                name="Caux6b[d%d,%d]"%(r+1,t+1))

    for r in R:
        m.addConstr(sum_var[r]==quicksum(theta[r,l] for l in leaf_nodes),
                            name="Caux7[%d]"%(r+1))

    for r in R:
        m.addGenConstrLog(sum_var[r], log_sumvar[r],name="Caux8[%d]"%(r+1))

    m.update()

    ################ SOLVE

    m.optimize()

    gap = m.MIPGap
    time = m.Runtime
    if m.Status == 3:
            m.computeIIS()
            m.write('infeasible_constraints.ilp')
    # Saving the solution
    pfs_x_sol = [0]*preds
    for j in P:
        pfs_x_sol[j]=pfs_x[j].x

    z_sol = np.zeros((n_trees,len(leaf_nodes)))
    beta_sol = np.zeros((n_trees,preds))

    for r in R:
        for l in leaf_nodes:
            z_sol[r,l-leaf_nodes[0]] = z[r,l].x
        for j in P:
            beta_sol[r,j] = beta[r,j].x


    return(pfs_x_sol, z_sol , beta_sol, time, gap)
