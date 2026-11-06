import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from scipy.special import comb
from tqdm import tqdm
from classical_algorithms import find_max_clique_rec

def find_max_clique_hybrid(graph: nx.Graph, grover="naive", grover_condition="first_level", **kwargs):
    return find_max_clique_rec(graph, set(), set(graph.nodes()), set(), 0, grover=grover, grover_condition=grover_condition, **kwargs)

if __name__ == "__main__":
    pass

    graph = nx.erdos_renyi_graph(20, 0.9)
    print(f"find_max_clique_rec: {find_max_clique_rec(graph, set(), set(graph.nodes()), set(), 0)}")
    print(f"find_max_clique_hybrid(naive): {find_max_clique_hybrid(graph, grover="naive")}")
    print(f"find_max_clique_hybrid(dicke): {find_max_clique_hybrid(graph, grover="dicke")}")
    print(f"find_max_clique_hybrid(dicke, min_density=0.8, max_density=0.95): {find_max_clique_hybrid(graph, 
                                                                                                      grover="dicke", 
                                                                                                      grover_condition="density", 
                                                                                                      min_density=0.93, 
                                                                                                      max_density=0.98,
                                                                                                      grover_max_n=15)}")

    for _ in tqdm(range(1000)):
        graph = nx.erdos_renyi_graph(25, 0.94)
        result1 = find_max_clique_rec(graph, set(), set(graph.nodes()), set(), 0)
        result2 = find_max_clique_hybrid(graph, 
                                grover="dicke", 
                                grover_condition="density", 
                                min_density=0.90, 
                                max_density=0.98,
                                grover_max_n=15)
        if result2[1] < result1[1]:
            print("FOUND!")
            print(result1)
            print(result2)