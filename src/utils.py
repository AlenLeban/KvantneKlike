import itertools
import math
import random
import numpy as np
import networkx as nx

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

def count_k_cliques(G, k):
    cliques = set()
    for maximal in nx.find_cliques(G):
        if len(maximal) >= k:
            for subset in itertools.combinations(sorted(maximal), k):
                cliques.add(subset)

    return len(cliques)

def generate_graph_with_k_clique(n, p, k):
    graph = nx.erdos_renyi_graph(n, p)
    clique_nodes = np.arange(n)
    np.random.shuffle(clique_nodes)
    if int(k) != k:
        percent = k - int(k)
        if random.random() < percent:
            k = int(k) + 1
    k = int(k)
    clique_nodes = clique_nodes[:k]
    # print(clique_nodes)
    edges_to_add = []
    for node1 in clique_nodes:
        for node2 in clique_nodes:
            if node1 == node2:
                continue
            if not graph.has_edge(node1, node2) and (node1, node2) not in edges_to_add:
                edges_to_add.append((node1, node2))
    graph.add_edges_from(edges_to_add)
    return {
        "graph": graph,
        "k": k
    }

def generate_k_clique_instance(size_info):
    return generate_graph_with_k_clique(size_info["n"], size_info["p"], size_info["k"])

def generate_max_clique_instance(size_info):
    return generate_graph_with_k_clique(size_info["n"], size_info["p"], size_info["k"])

def generate_random_graph_instance(size_info):
    return {
        "graph": nx.erdos_renyi_graph(size_info["n"], size_info["p"])
    }

def validate_max_clique_solutions(problem_instance, solutions):
    max_clique_size = 0
    maximal_clique_counts = dict()
    for clique in nx.algorithms.clique.find_cliques(problem_instance["graph"]):
        size = len(clique)
        if size > max_clique_size:
            max_clique_size = size
        maximal_clique_counts[size] = maximal_clique_counts.get(size, 0) + 1
    # print(len(max_clique_nodes))
    validation = validate_k_clique_solutions({
            "graph": problem_instance["graph"],
            "k": max_clique_size}
        , solutions)
    validation["maximal_clique_counts"] = maximal_clique_counts
    return validation

def validate_k_clique_solutions(problem_instance, solutions):
    valid_cliques = 0
    k_cliques = 0
    is_size_k = 0
    k = problem_instance["k"]
    for solution in solutions:
        is_valid = is_clique(problem_instance["graph"], solution)
        is_k_clique = sum(solution) == k and is_valid
        valid_cliques += 1 if is_valid else 0
        k_cliques += 1 if is_k_clique else 0
        is_size_k += 1 if sum(solution) == k else 0

    k_clique_count = count_k_cliques(problem_instance["graph"], k)

    return {
        "valid_cliques": valid_cliques,
        "k": k,
        "is_size_k": is_size_k,
        "k_cliques": k_cliques,
        "found_solution": k_cliques > 0,
        "k_nodes_clique_percent": k_clique_count / math.comb(problem_instance["graph"].number_of_nodes(), k),
    }

def is_clique(graph: nx.Graph, solution):
    solution_npy = np.array(solution)
    mask = solution_npy == 1
    node_indices = np.array(list(graph.nodes()))[mask]
    for n1 in node_indices:
        for n2 in node_indices:
            if n1 == n2:
                continue
            if not graph.has_edge(n1, n2):
                return False
    return True

def er_max_clique_size(n, p):
    return float(2*np.log(n)/np.log(1/p))

if __name__ == "__main__":

    print(er_max_clique_size(10, 0.4))