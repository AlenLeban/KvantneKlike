from utils import validate_k_clique_solutions, validate_max_clique_solutions
import networkx as nx
from matplotlib import pyplot as plt
from itertools import product
import pylcs

def helper_validate_max_clique_solutions(g, x, p):
    return validate_max_clique_solutions(g, x)

def helper_validate_k_clique_solutions(g, x, p):
    return validate_k_clique_solutions(g, x)

def generate_d_deletion_graph(b, n, d):
    graph = nx.Graph()
    codewords = product(range(b), repeat=n)
    codeword_strings = [''.join(str(val) for val in c) for c in codewords]
    print(codeword_strings)
    graph.add_nodes_from(codeword_strings)
    edges = []
    plt.figure()
    for i in range(len(codeword_strings)):
        for j in range(i+1, len(codeword_strings)):
            s1 = codeword_strings[i]
            s2 = codeword_strings[j]
            if pylcs.lcs_sequence_length(s1, s2) < n - d:
                edges.append((s1, s2))

    graph.add_edges_from(edges)

    nx.draw(graph, with_labels=True)
    plt.show()

if __name__ == "__main__":
    generate_d_deletion_graph(3, 4, 2)