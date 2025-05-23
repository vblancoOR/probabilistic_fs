# Auxiliar functions for PFS
import numpy as np
import pandas as pd
import random as rd
from tqdm.auto import tqdm

import pfs_model_max as pfs_1
import pfs_model_min as pfs_2
import pfs_model_kappa as pfs_3

###
def create_ordered_nodes(tree,tree_depth):
    # Returns the tree nodes with their information ordered by depth.
    # tree: DecisionTreeClassifier
    # tree_depth
    depth = tree_depth + 1
    depths = tree.tree_.compute_node_depths()
    features = tree.tree_.feature
    thresholds = tree.tree_.threshold
    expanded = [0]*len(features)

    ordered_nodes = [[x,y,z,t] for (x,y,z,t) in zip(depths,features,thresholds,expanded)]

    while len(ordered_nodes) != 2**depth -1:
        for i in range(len(ordered_nodes)):
            if ordered_nodes[i][0] != depth and ordered_nodes[i][1] == -2 and ordered_nodes[i][3] == 0:
                ordered_nodes[i][3] = 1
                ordered_nodes.insert(i+1,[ordered_nodes[i][0]+1,-2,-2.0,0])
                ordered_nodes.insert(i+2,[ordered_nodes[i][0]+1,-2,-2.0,0])

    ordered_nodes.sort(key = lambda x: x[0])

    return(ordered_nodes)

###
def create_leaves(tree,tree_depth):
    # Returns the full list of leaves with their classes.
    # tree: DecisionTreeClassifier
    # tree_depth
    depth = tree_depth + 1
    props = tree.tree_.value
    depths = tree.tree_.compute_node_depths()
    features = tree.tree_.feature
    expanded = [0]*len(props)

    pre_leaves = [[x,y,z,t] for (x,y,z,t) in zip(depths,features,props,expanded)]

    while len(pre_leaves) != 2**(depth)-1:
        for i in range(len(pre_leaves)):
            if pre_leaves[i][0] != depth and pre_leaves[i][1] == -2 and pre_leaves[i][3] == 0:
                pre_leaves[i][3] = 1
                pre_leaves.insert(i+1,[pre_leaves[i][0]+1,-2,pre_leaves[i][2],0])
                pre_leaves.insert(i+2,[pre_leaves[i][0]+1,-2,pre_leaves[i][2],0])

    pre_leaves.sort(key = lambda x: x[0])
    leaves = []

    for j in range(len(pre_leaves)):
        if pre_leaves[j][0] == depth:
            if pre_leaves[j][2][0][0] > pre_leaves[j][2][0][1]:
                leaf_class = 0
            else:
                leaf_class = 1
            leaves.append(leaf_class)

    return(leaves)

###
def vars_and_vals(ordered_nodes, tree_depth):
    # Returns the full lists of features and threshold values over the ordered_nodes.
    # ordered_nodes
    # tree_depth
    vals = []
    features = []
    depth = tree_depth + 1

    for j in range(len(ordered_nodes)):
        if ordered_nodes[j][0] == depth:
            continue

        if ordered_nodes[j][1] == -2:
            features.append(0)
            vals.append(-3)
        else:
            features.append(ordered_nodes[j][1])
            vals.append(ordered_nodes[j][2])

    return(features, vals)

###

# For the following functions, variables_information is a list of dimension p.
# Each element indicates wheter variable x_p is inmutable (value 0),
# continuous and positively correlated with variable y (value 1),
# continuous and negatively correlated with variable y (value -1),
# or binary (value -2).
def calculate_deviations(data, variables_information, max_p = 0.8):
    # Returns the lists of max_allowed deviations for predictor variables,
    # with and without effort, necessary for probability estimations.
    # data: full train dataset.
    # variables_information
    # max_p: upper bound of change probability for binary predictor variables.
    P = range(data.shape[1])
    desv = []
    desv_eff = []

    for j in P:
        if variables_information[j] == 0:
            desv.append(0)
            desv_eff.append(0)

        if abs(variables_information[j]) == 1:
            desv.append(np.sqrt(np.var(data.iloc[:,j])))
            if variables_information[j]>0:
                desv_eff.append(1.5*np.sqrt(np.var(data.iloc[:,j])))
            else:
                desv_eff.append(-1.5*np.sqrt(np.var(data.iloc[:,j])))

        if variables_information[j] == -2:
            p = sum(data.iloc[:,j])/data.shape[0]
            flip_coin = max(p, 1 - p)
            desv.append(flip_coin)
            desv_eff.append(max(flip_coin-(np.sqrt(p*(1-p))/2), max_p))

    return desv, desv_eff


###
def calculate_probs(indiv_x, tree_depth, ordered_nodes, variables_information,
                    desv, n_iters = 1000):
    # Returns the estimated probability of change over n_iter simulations for
    # an individual in the sample in the ordered_nodes of a DecisionTreeClassifier.
    # indiv_x: predictor variables of the infividual in the dataset
    # tree_depth
    # ordered_nodes
    # variables_information
    # desv: list of deviations without effort
    # n_iter: number of scenarios to simulate
    probs_ind = []
    depth = tree_depth + 1

    for j in range(len(ordered_nodes)):
        count = 0

        if ordered_nodes[j][0] == depth:
            continue

        if variables_information[ordered_nodes[j][1]] != -2:
            for k in range(n_iters):
                rd.seed(k)
                if rd.random() < 0.5:
                    sign = -1
                else:
                    sign = 1

                rd.seed(k)
                epsilon = sign*(rd.random())*desv[ordered_nodes[j][1]]
                value = indiv_x.iloc[ordered_nodes[j][1]] + epsilon

                if value >= ordered_nodes[j][2]:
                    count += 1

        else:
            for k in range(n_iters):
                rd.seed(k)
                if rd.random() > desv[ordered_nodes[j][1]]:
                    value = 1 - indiv_x.iloc[ordered_nodes[j][1]]
                else:
                    value = indiv_x.iloc[ordered_nodes[j][1]]

                if value > ordered_nodes[j][2]:
                    count += 1

        probs_ind.append(count/n_iters)

    return(probs_ind)

###
def calculate_probs_eff(indiv_x, tree_depth, ordered_nodes,
                        variables_information, desv_eff, n_iters = 1000):
    # Returns the estimated probability of change with effort over n_iter simulations for
    # an individual in the sample in the ordered_nodes of a DecisionTreeClassifier.
    # indiv_x: predictor variables of the infividual in the dataset
    # tree_depth
    # ordered_nodes
    # variables_information
    # desv_eff: list of deviations with effort
    # n_iter: number of scenarios to simulate
    probs_eff_ind = []
    depth = tree_depth + 1

    for j in range(len(ordered_nodes)):
        count = 0

        if ordered_nodes[j][0] == depth:
            continue

        if variables_information[ordered_nodes[j][1]] != -2:
            for k in range(n_iters):

                rd.seed(k)
                epsilon = rd.random()*desv_eff[ordered_nodes[j][1]]
                value = indiv_x.iloc[ordered_nodes[j][1]] + epsilon
                if value >= ordered_nodes[j][2]:
                    count += 1

        else:
            for k in range(n_iters):
                rd.seed(k)
                if rd.random() > desv_eff[ordered_nodes[j][1]]:
                    value = 0
                else:
                    value = indiv_x.iloc[ordered_nodes[j][1]]

                if value > ordered_nodes[j][2]:
                    count += 1

        probs_eff_ind.append(count/n_iters)

    return(probs_eff_ind)

###
def build_forest(forest, n_trees, depth):
    # Returns the RandomForestClassifier information in the appropriate input
    # format for the MINLP model.
    # forest: RandomForestClassifier
    # n_trees: number of trees in the forest
    # depth

    leaves = []
    tree_vars_ = []
    tree_vals_ = []

    for i in range(n_trees):
        tree = forest.estimators_[i]
        tree_leaves = create_leaves(tree, depth)
        leaves.append(tree_leaves)

        ordered_nodes = create_ordered_nodes(tree, depth)
        tree_vars_.append(vars_and_vals(ordered_nodes, depth)[0])
        tree_vals_.append(vars_and_vals(ordered_nodes, depth)[1])

    leaves_final = pd.DataFrame(np.array(leaves))
    vars_final = pd.DataFrame(np.array(tree_vars_))
    vals_final = pd.DataFrame(np.array(tree_vals_))

    return(leaves_final, vars_final, vals_final)

###
def calculate_forest_probs(individual_x, forest, n_trees, depth, desv, desv_eff,
                           variables_information, n_iters = 1000):
    # Returns the estimated probabilities of change, with and without effort,
    # of an individual in the RandomForestClassifier in the appropriate format
    # for the MINLP problem.
    # indiv_x: predictor variables of the infividual in the dataset
    # forest: RandomForestClassifier
    # n_trees
    # depth
    # desv: list of deviations without effort
    # desv_eff: list of deviations with effort
    # variables_information
    # n_iter: number of scenarios to simulate
    probs = []
    probs_eff = []

    for i in range(n_trees):
        tree = forest.estimators_[i]
        ordered_nodes = create_ordered_nodes(tree, depth)
        probs.append(calculate_probs(individual_x, depth, ordered_nodes,
                     variables_information, desv))
        probs_eff.append(calculate_probs_eff(individual_x, depth, ordered_nodes,
                         variables_information, desv_eff))

    probs_final = pd.DataFrame(np.array(probs))
    probs_eff_final = pd.DataFrame(np.array(probs_eff))

    return(probs_final, probs_eff_final)

###
def pfs_ranking(X_train, y_train, variables_information, rf, n_trees, depth, C,
                timelimit=300, model=3, desired_class=1, n_iter=1000,
                kappa=0.5, epsilon=0.000001):
    # Returns the variable importance ranking following the PFS approach
    # X_train: train dataset predictor variables
    # y_train: train dataset target variable
    # variables information
    # rf: RandomForestClassifier
    # n_trees
    # depth
    # C: number of variables allowed with effort
    # timelimit
    # Model: 1(max-path) - 2 (min-path) - 3 (kappa-path)
    # desired_class
    # n_iter
    # epsilon: constant involved in constraints (2) of the MINLP
    # kappa: desired value for the kappa-path problem
    # n_paths: value for kappa-path problem
    # mu_path: value for kappa-path problem
    n_paths = int(kappa*(2**depth))
    mu_path = 10**-(depth + 1)

    desv, desv_eff = calculate_deviations(X_train, variables_information)
    forest_leaves, forest_vars, forest_vals = build_forest(rf, n_trees, depth)

    zero_class_count = sum(1 for i in y_train if i == 0)

    total_used_vars = np.zeros(X_train.shape[1])

    for i in tqdm(range(len(X_train))):

        indiv = pd.DataFrame(X_train.iloc[i,:]).transpose()
        indiv["y"] = y_train.iloc[i]

        if y_train.iloc[i] == 0:
            probs, probs_eff = calculate_forest_probs(X_train.iloc[i,:], rf,
                                                      n_trees, depth,
                                                      desv, desv_eff,
                                                      variables_information,
                                                      n_iter)
            if model==1:
                pfs_sol, z_sol , beta_sol, time, gap = pfs_1.pfs(indiv, C,
                                                       probs, probs_eff,
                                                       forest_vars, forest_vals,
                                                       depth, forest_leaves,
                                                       timelimit,
                                                       desired_class, epsilon
                                                       )


            if model==2:
                pfs_sol, z_sol , beta_sol, time, gap = pfs_1.pfs_min(indiv, C,
                                                       probs, probs_eff,
                                                       forest_vars, forest_vals,
                                                       depth, forest_leaves,
                                                       timelimit,
                                                       desired_class, epsilon
                                                       )
            if model==3:
                pfs_sol, z_sol , beta_sol, time, gap = pfs_3.pfs_kappa(indiv, C,
                                                       n_paths, mu_path,
                                                       probs, probs_eff,
                                                       forest_vars, forest_vals,
                                                       depth, forest_leaves,
                                                       desired_class, epsilon,
                                                       timelimit)
            beta_vars = np.sum(beta_sol,axis=0)
            used_vars = []

            for i in range(len(beta_vars)):
                if beta_vars[i] > 0:
                    used_vars.append(1)
                else:
                    used_vars.append(0)

            total_used_vars += np.array(used_vars)

    ranking = pd.DataFrame(total_used_vars/zero_class_count).transpose()
    ranking.columns = X_train.columns

    mask = np.array(variables_information) == 0
    ranking.iloc[0, mask] = 0

    sorted_idx = ranking.values[0].argsort()
    sorted_columns = ranking.columns[sorted_idx]
    ordered_ranking = ranking[sorted_columns]

    selec_vars = ordered_ranking.columns[::-1][:C]

    vars_to_modify = ranking.columns.get_indexer(selec_vars)

    list_vars_eff = np.zeros(X_train.shape[1])
    list_vars_eff[vars_to_modify] = 1

    return ordered_ranking, ranking, list(list_vars_eff)
