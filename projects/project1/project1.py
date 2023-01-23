#######################
###     IMPORTS     ###
#######################
import sys
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

#######################
###   GLOBAL VARS   ###
#######################
# VAR NAME:  # DESCRIPTION:
MIN_VAL = 1  # minimum value any variable in the graph can assume


def init_graph(inputfilename):
    """
    Initializes a DAG, maximum values dictionary, idx2names dict, and dataframe from the data found at 'inputfilename'.
    """
    # read in the data file:
    df = pd.read_csv(inputfilename)
    # determine max value for each variable in the network:
    max_vals = df.max().to_dict()
    # initialize: an empty (directed) graph, a dict. mapping node indices to variable names:
    g, idx2names = nx.DiGraph(), dict()
    # for each variable in the data file, init. a graph node and log its index-name association:
    for idx, var_name in enumerate(list(max_vals.keys())):
        g.add_node(idx)
        idx2names[idx] = var_name
    # return the initialized graph, max_vals dict, idx2names dict, and dataframe:
    return g, idx2names, max_vals, df


def write_gph(dag, idx2names, filename):
    """
    A utility function which outputs the structure of 'dag' using variable names (specified by idx2names) to 'filename' in HRF.
    """
    with open(filename, "w") as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def generate_neighbors(g: nx.Graph):
    """
    Generates and returns the edge sets of all DAGs which are within one operation of the supplied DAG, g.
    An operation can be an: edge addition; edge deletion; edge reversal; on a **SINGLE** edge.
    """
    # retrieve the current set of nodes & edges:
    source_node_set, source_edge_set = set(g.nodes()), set(g.edges())
    # init. the ret. val:
    neighbor_edge_sets = []

    # first set of ret vals: graphs within one edge addition of g
    for node in source_node_set:
        # rule: for a node n, you cannot add an arc from n to any of n's ancestor or n itself without creating a cycle.
        node_ancestors = nx.ancestors(g, node).union({node})
        valid_new_neighbors = source_node_set - node_ancestors

        for new_neighbor in valid_new_neighbors:
            # omit any neighbors which result from adding edges which already exist in the source graph:
            if (node, new_neighbor) not in source_edge_set:
                neighbor_edge_set = source_edge_set.union({(node, new_neighbor)})
                neighbor_edge_sets.append(neighbor_edge_set)

    # second set of ret vals: graphs within one edge deletion of g
    for edge in source_edge_set:
        neighbor_edge_set = source_edge_set.copy()
        neighbor_edge_set.remove(edge)
        neighbor_edge_sets.append(neighbor_edge_set)

    # third (and final) set of ret vals: graphs within one edge reversal of g
    for edge in source_edge_set:
        neighbor_edge_set = source_edge_set.copy()
        neighbor_edge_set.remove(edge)
        neighbor_edge_set.add(edge[::-1])

        # initialize a new graph with the same nodes as the source DAG & one edge reversed:
        potential_neighbor_graph = nx.DiGraph(incoming_graph_data=neighbor_edge_set)
        potential_neighbor_graph.add_nodes_from(source_node_set)

        # check if the neighboring graph is a DAG:
        if nx.is_directed_acyclic_graph(potential_neighbor_graph):
            neighbor_edge_sets.append(set(potential_neighbor_graph.edges()))

    return neighbor_edge_sets


def bayesian_score(g_edge_set: set, data: pd.DataFrame):
    """
    Returns the Bayesian score for the DAG specified by g_edge_set over the samples in 'data'.
    """
    return 0


def hill_climb_search(g: nx.Graph, data: pd.DataFrame):
    # initialize ret val & search parameters:
    est_dag, curr_score, score_improvement = g, bayesian_score(g, data), np.Inf

    # loop until convergence:
    while score_improvement > 0:
        # gen. a list of the edge sets of all DAGs within one operation of our current DAG:
        neighbor_dag_edge_sets = generate_neighbors(est_dag)

        # score and sort the neighboring DAGs by Bayesian score:
        neighbor_dag_and_scores = [
            (bayesian_score(edge_set, data), edge_set)
            for edge_set in neighbor_dag_edge_sets
        ].sort(key=lambda a: a[0])

        # retrieve the neighbor with the highest Bayesian score:
        best_neighbor_edge_set, best_neighbor_score = (
            neighbor_dag_and_scores[0][0],
            neighbor_dag_and_scores[0][1],
        )

        # check to see if there's a score improvement:
        score_improvement = best_neighbor_score - curr_score

        # if there's a score improvement, init. another search iteration with the improved DAG:
        if score_improvement > 0:
            best_alt_dag = nx.DiGraph(incoming_graph_data=best_neighbor_edge_set)
            best_alt_dag.add_nodes_from(set(est_dag.nodes()))
            est_dag, curr_score = best_alt_dag, best_neighbor_score

    # return the (local) optimal DAG:
    return est_dag


def main():
    # argument parsing:
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    inputfilename, outputfilename = sys.argv[1], sys.argv[2]
    # initialize a graph with a node for each variable and no edges:
    g, idx2names, max_vals, df = init_graph(inputfilename)
    # estimate a network structure using hill climb search:
    est_dag = hill_climb_search(g, df)
    # output the learned DAG:
    write_gph(est_dag, idx2names, outputfilename)


if __name__ == "__main__":
    main()
