from utils import validate_k_clique_solutions, validate_max_clique_solutions
import networkx as nx
from matplotlib import pyplot as plt
from itertools import combinations

def helper_validate_max_clique_solutions(g, x, p):
    return validate_max_clique_solutions(g, x)

def helper_validate_k_clique_solutions(g, x, p):
    return validate_k_clique_solutions(g, x)

def generat_d_deleteion_graph(b, n, d):
    graph = nx.Graph()
    combs = combinations(range(0, b), n)
    plt.figure()
    nx.draw(graph, with_labels=True)
