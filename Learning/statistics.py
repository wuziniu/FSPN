from collections import Counter

from Structure.nodes import get_nodes_by_type, Sum, Product, Factorize, Leaf, get_number_of_edges, get_depth, Node, bfs
import logging

logger = logging.getLogger(__name__)


def get_structure_stats_dict(node):
    nodes = get_nodes_by_type(node, Node)
    num_nodes = len(nodes)

    node_types = dict(Counter([type(n) for n in nodes]))

    edges = get_number_of_edges(node)
    layers = get_depth(node)

    params = 0
    for n in nodes:
        if isinstance(n, Sum):
            params += len(n.children)
        if isinstance(n, Leaf):
            params += len(n.parameters)

    result = {"nodes": num_nodes, "params": params, "edges": edges, "layers": layers, "count_per_type": node_types}
    return result


def get_structure_stats(node):
    num_nodes = len(get_nodes_by_type(node, Node))
    sum_nodes = get_nodes_by_type(node, Sum)
    n_sum_nodes = len(sum_nodes)
    n_prod_nodes = len(get_nodes_by_type(node, Product))
    n_fact_nodes = len(get_nodes_by_type(node, Factorize))
    leaf_nodes = get_nodes_by_type(node, Leaf)
    n_leaf_nodes = len(leaf_nodes)
    edges = get_number_of_edges(node)
    layers = get_depth(node)
    params = 0
    for n in sum_nodes:
        params += len(n.children)
    #for l in leaf_nodes:
     #   params += len(l.parameters)


    return """---Structure Statistics---
# nodes               %s
    # sum nodes       %s
    # factorize nodes %s
    # prod nodes      %s
    # leaf nodes      %s
# params              %s
# edges               %s
# layers              %s""" % (
        num_nodes,
        n_sum_nodes,
        n_fact_nodes,
        n_prod_nodes,
        n_leaf_nodes,
        params,
        edges,
        layers,
    )

def get_range_states(node):
    def print_range(n):
        if isinstance(n, Leaf):
            print(n.range)

    bfs(node, print_range)
    return None

def get_scope_states(node):
    def print_scope(n):
        if isinstance(n, Leaf):
            print(n.scope)

    bfs(node, print_scope)
    return None
