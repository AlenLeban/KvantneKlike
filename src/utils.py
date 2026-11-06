import itertools
import math
import random
import numpy as np
import networkx as nx
import json
import pylcs
from scipy.special import comb

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

def count_k_cliques(G, k):
    if k == 0:
        return 0
    cliques = set()
    for maximal in nx.find_cliques(G):
        if len(maximal) >= k:
            for subset in itertools.combinations(sorted(maximal), k):
                cliques.add(subset)

    return len(cliques)

def has_k_clique(G, k):
    if k == 0:
        return False
    cliques = set()
    for maximal in nx.find_cliques(G):
        if len(maximal) >= k:
            return True
    return False

def generate_graph_with_k_clique(n, p, k):
    graph = nx.erdos_renyi_graph(n, p)
    if k < 0:
        max_k = 0
        for clique in nx.find_cliques(graph):
            max_k = max(max_k, len(clique))
        k = max_k + k
    else:
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


def generate_kcore_graph_instance(size_info):
    p = size_info["p"]
    while True:
        graph = nx.erdos_renyi_graph(size_info["n"], p)
        # max_clique_size = 0
        # for clique in nx.algorithms.clique.find_cliques(graph):
        #     size = len(clique)
        #     if size > max_clique_size:
        #         max_clique_size = size
        k_core = nx.k_core(graph, size_info["k_score_size"])
        if k_core.number_of_nodes() == size_info["n"]:        
            return {
                "graph": k_core
            }
        else:
            p += 0.01

def generate_random_instance_from_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    random_instance = data[random.randint(0, len(data)-1)]
    graph = nx.parse_adjlist(random_instance["graph"])
    return {
        "graph": graph
    }

def generate_specific_instance_from_file(filepath, instance_index):
    with open(filepath, "r") as f:
        data = json.load(f)
    random_instance = data[instance_index]
    graph = nx.parse_adjlist(random_instance["graph"])
    return {
        "graph": graph
    }

def generate_config_model_graph_instance(size_info):
    graph = nx.configuration_model([size_info["k"]]*size_info["n"], create_using=nx.Graph)
    return {
        "graph": graph
    }

def generate_d_deletion_graph(b, n, d, include_all_lengths=False):
    graph = nx.Graph()
    codewords = itertools.product(range(b), repeat=n)
    codeword_strings = [''.join(str(val) for val in c) for c in codewords]
    if include_all_lengths:
        for i in range(n-1):
            codewords2 = itertools.product(range(b), repeat=n-i-1)
            codeword_strings2 = [''.join(str(val) for val in c) for c in codewords2]
            codeword_strings.extend(codeword_strings2)
    # print(codeword_strings)
    graph.add_nodes_from(codeword_strings)
    edges = []
    # plt.figure()
    for i in range(len(codeword_strings)):
        for j in range(i+1, len(codeword_strings)):
            s1 = codeword_strings[i]
            s2 = codeword_strings[j]
            dlcs = (len(s1) + len(s2) - 2 * pylcs.lcs_sequence_length(s1, s2)) / 2
            if dlcs >= d + 1:
                edges.append((s1, s2))

    graph.add_edges_from(edges)

    # nx.draw(graph, with_labels=True)
    # plt.show()

    return {
        "graph": graph
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
        "n": problem_instance["graph"].number_of_nodes(),
        "m": problem_instance["graph"].number_of_edges(),
        "valid_cliques": valid_cliques,
        "k": k,
        "is_size_k": is_size_k,
        "k_cliques": k_cliques,
        "found_solution": k_cliques > 0,
        "k_nodes_clique_percent": k_clique_count / math.comb(problem_instance["graph"].number_of_nodes(), k),
    }

def is_clique(graph: nx.Graph, solution):
    solution_npy = np.array(solution)
    # print(solution_npy)
    mask = solution_npy == 1
    # print(mask)
    # print(sorted(np.array(list(graph.nodes()))))
    node_indices = np.array(sorted(list(graph.nodes())))[mask]
    # print(node_indices)
    for n1 in node_indices:
        for n2 in node_indices:
            if n1 == n2:
                continue
            if not graph.has_edge(n1, n2):
                return False
    return True

def er_max_clique_size(n, p):
    return float(2*np.log(n)/np.log(1/p))

def graph_info(g: nx.Graph) -> dict:
    max_clique_size = 0
    maximal_clique_counts = dict()
    for clique in nx.algorithms.clique.find_cliques(g):
        size = len(clique)
        if size > max_clique_size:
            max_clique_size = size
        maximal_clique_counts[size] = maximal_clique_counts.get(size, 0) + 1

    smaller_clique_counts = []
    for k in range(max_clique_size, max(max_clique_size-3, 0), -1):
        smaller_clique_counts.append(count_k_cliques(g, k))

    return {
        "number_of_nodes": g.number_of_nodes(),
        "number_of_edges": g.number_of_edges(),
        "max_clique_size": max_clique_size,
        "max_clique_count": smaller_clique_counts[0],
        "smaller_clique_counts": smaller_clique_counts[1:]
    }

def grover_mc_oracles_naive(n, correct_max_k, num_solutions, min_k=2): #TODO: Add steps for estimating num_solutions
    steps = 0
    for k in range(min_k, correct_max_k+1):
        to_add = np.pi/4 * np.sqrt((2**n) / num_solutions)
        # print(f"Add: {to_add}")
        steps += to_add
    return int(steps)

def grover_mc_oracles_dicke(n, correct_max_k, num_solutions, min_k=2):
    steps = 0
    for k in range(min_k, correct_max_k+1):
        to_add = np.pi/4 * np.sqrt((comb(n, k)) / num_solutions)
        # print(f"Add: {to_add}")
        steps += to_add
    return int(steps)

if __name__ == "__main__":
    # g = nx.erdos_renyi_graph(8, 0.5)
    # print(graph_info(g))
    pass
