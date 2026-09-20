from utils import validate_k_clique_solutions, validate_max_clique_solutions
import networkx as nx
from matplotlib import pyplot as plt
from itertools import product
import pylcs

def helper_validate_max_clique_solutions(g, x, p):
    return validate_max_clique_solutions(g, x)

def helper_validate_k_clique_solutions(g, x, p):
    return validate_k_clique_solutions(g, x)

# if __name__ == "__main__":
#     g = generate_d_deletion_graph(4, 2, 1, include_all_lengths=False)
