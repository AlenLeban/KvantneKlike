import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from tqdm import tqdm
from utils import count_k_cliques, grover_mc_oracles_naive, grover_mc_oracles_dicke

def find_max_clique_rec(graph: nx.Graph, P: set, N: set, Q: set, max_size: int, **kwargs):
    candidates = N.difference(Q)
    total_expanded = 0
    if len(candidates) == 0:
        return len(P), 0
    if len(candidates) + len(P) <= max_size:
        return len(P), 0
    newly_added_q = []
    for candidate in candidates:
        if candidate not in Q:
            N_new = N.intersection(graph.neighbors(candidate))
            total_expanded += 1
            P_new = P.union({candidate})
            found_size, expanded = find_max_clique_rec(graph, P_new, N_new, Q, max_size)

            # check for grover
            should_run_grover = "grover" in kwargs and kwargs.get("grover") != None
            
            induced_subgraph = None
            
            if "grover_condition" in kwargs and should_run_grover:
                grover_condition = kwargs.get("grover_condition")
                if grover_condition == "first_level":
                    should_run_grover = True

                elif grover_condition == "density":
                    if "min_density" not in kwargs:
                        print("For grover condition ''density'', min_density needs to be specified in kwargs!")
                        return 0, 0
                    if "max_density" not in kwargs:
                        print("For grover condition ''density'', max_density needs to be specified in kwargs!")
                        return 0, 0
                    min_density = kwargs.get("min_density")
                    max_density = kwargs.get("max_density")
                    induced_subgraph_nodes = N_new.union([candidate])
                    induced_subgraph = nx.induced_subgraph(graph, induced_subgraph_nodes)
                    subgraph_density = nx.density(induced_subgraph)
                    if not(min_density <= subgraph_density <= max_density):
                        should_run_grover = False

            if "grover_max_n" in kwargs:
                if kwargs.get("grover_max_n") < len(N_new.union([candidate])):
                    should_run_grover = False

            if should_run_grover:
                if induced_subgraph == None:
                    induced_subgraph = nx.induced_subgraph(graph, N_new.union([candidate]))
                method = kwargs.get("grover")
                # print(induced_subgraph)
                if method == "naive":
                    subgraph_max_clique_size = find_max_clique_rec(induced_subgraph, set(), set(induced_subgraph.nodes()), set(), 0)[0]
                    grover_steps = grover_mc_oracles_naive(
                        induced_subgraph.number_of_nodes(), 
                        subgraph_max_clique_size, 
                        count_k_cliques(induced_subgraph, subgraph_max_clique_size),
                        min_k=len(P_new))
                elif method == "dicke":
                    subgraph_max_clique_size = find_max_clique_rec(induced_subgraph, set(), set(induced_subgraph.nodes()), set(), 0)[0]
                    grover_steps = grover_mc_oracles_dicke(
                        induced_subgraph.number_of_nodes(), 
                        subgraph_max_clique_size, 
                        count_k_cliques(induced_subgraph, subgraph_max_clique_size))
                else:
                    print(f"Method {method} is invalid!")
                    return 0, 0
                # print(grover_steps)
                total_expanded += grover_steps
            else:
                total_expanded += expanded

            if found_size > max_size:
                max_size = found_size
            if candidate not in Q:
                Q.add(candidate)
                newly_added_q.append(candidate)
    for q in newly_added_q:
        Q.remove(q)
    return max_size, total_expanded


if __name__ == "__main__":


    # correctness check
    for i in tqdm(range(400)):
        graph = nx.erdos_renyi_graph(40, 0.8, seed=i)

        # nx.draw(graph, with_labels=True)
        # plt.show()

        max_size, expanded = find_max_clique_rec(graph, set(), set(graph.nodes()), set(), 0)
        print(f"Expanded: {expanded}")
        # correct_size = len(get_max_clique(graph))
        # if max_size != correct_size:
        #     print("WRONG")
        #     print(f"Found max size: {max_size}")
        #     print(f"Correct size: {correct_size}")
        #     print(f"seed: {i}")

