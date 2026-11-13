import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from tqdm import tqdm
from max_clique_github.max_clique.main import get_max_clique
from utils import count_k_cliques, grover_mc_oracles_naive, grover_mc_oracles_dicke

def find_max_clique_rec(graph: nx.Graph, P: set, N: set, Q: set, max_size: int, **kwargs):

    used_grover = False
    total_expanded = 0
    total_grover = 0
    # if n_ordering == None:
    #     degree_node_pairs = []
    #     for n in N:
    #         degree = graph.degree(n)
    #         degree_node_pairs.append((n, degree))
    #     n_ordering = list(map(lambda x: x[1], sorted(degree_node_pairs)))


    candidates = N.difference(Q)
    if len(candidates) == 0:
        return len(P), (0, 0), False
    if len(candidates) + len(P) <= max_size:
        return len(P), (0, 0), False
    newly_added_q = []
    sorted_candidates = list(map(lambda x: x[1], sorted([(graph.degree(c), c) for c in candidates], reverse=True)))
    # print(candidates)
    # print(sorted([(graph.degree(c), c) for c in candidates], reverse=True))
    # print(sorted_candidates)
    for candidate in sorted_candidates:
        if candidate not in Q:
            N_new = N.intersection(graph.neighbors(candidate))
            P_new = P.union({candidate})
            # check for grover first
            should_run_grover = "grover" in kwargs and kwargs.get("grover") != None
            induced_subgraph = None
            
            if "grover_condition" in kwargs and should_run_grover:
                grover_condition = kwargs.get("grover_condition")
                for condition in grover_condition:
                    
                    if condition == "density":
                        if "min_density" not in kwargs:
                            print("For grover condition ''density'', min_density needs to be specified in kwargs!")
                            return 0, (0, 0), False
                        if "max_density" not in kwargs:
                            print("For grover condition ''density'', max_density needs to be specified in kwargs!")
                            return 0, (0, 0), False
                        min_density = kwargs.get("min_density")
                        max_density = kwargs.get("max_density")
                        induced_subgraph_nodes = N_new.union(P_new)
                        induced_subgraph = nx.induced_subgraph(graph, induced_subgraph_nodes)
                        subgraph_density = nx.density(induced_subgraph)
                        if not(min_density <= subgraph_density <= max_density):
                            should_run_grover = False

                    elif condition == "min_degree":
                        if "min_degree_percent" not in kwargs:
                            print("For grover condition ''min_degree'', min_degree_percent needs to be specified in kwargs!")
                            return 0, (0, 0), False
                        min_degree_percent = kwargs.get("min_degree_percent")
                        induced_subgraph_nodes = N_new.union(P_new)
                        induced_subgraph = nx.induced_subgraph(graph, induced_subgraph_nodes)
                        min_degree = min([induced_subgraph.degree(n) for n in induced_subgraph_nodes])
                        max_degree = max([induced_subgraph.degree(n) for n in induced_subgraph_nodes])
                        if min_degree / max_degree < min_degree_percent:
                            should_run_grover = False

                    elif condition == "min_best_size":
                        if "min_best_size_num" not in kwargs:
                            print("For grover condition ''min_best_size_num'', min_best_size needs to be specified in kwargs!")
                            return 0, (0, 0), False
                        min_best_size_num = kwargs.get("min_best_size_num")
                        if min_best_size_num > max_size:
                            should_run_grover = False

            if "grover_max_n" in kwargs and should_run_grover:
                if kwargs.get("grover_max_n") < len(N_new.union([candidate])):
                    should_run_grover = False

            if should_run_grover:
                if induced_subgraph == None:
                    induced_subgraph = nx.induced_subgraph(graph, N_new.union(P_new))
                method = kwargs.get("grover")
                # print(induced_subgraph)
                min_k = len(P_new) + (1 if len(N_new) > 0 else 0)
                if method == "naive":
                    subgraph_max_clique_size = find_max_clique_rec(induced_subgraph, set(), set(induced_subgraph.nodes()), set(), 0)[0]
                    grover_steps = grover_mc_oracles_naive(
                        induced_subgraph,
                        induced_subgraph.number_of_nodes(), 
                        subgraph_max_clique_size, 
                        min_k=min_k)
                elif method == "dicke":
                    subgraph_max_clique_size = find_max_clique_rec(induced_subgraph, set(), set(induced_subgraph.nodes()), set(), 0)[0]
                    grover_steps = grover_mc_oracles_dicke(
                        induced_subgraph,
                        induced_subgraph.number_of_nodes(), 
                        subgraph_max_clique_size, 
                        min_k=min_k)
                else:
                    print(f"Method {method} is invalid!")
                    return 0, (0, 0), False
                # print(grover_steps)
                total_grover += grover_steps
                used_grover = True

            found_size, (expanded, grover_expanded), new_used_grover = find_max_clique_rec(graph, P_new, N_new, Q, max_size)

            if not should_run_grover:
                total_expanded += 1
                total_grover += grover_expanded
                total_expanded += expanded

            if found_size > max_size:
                max_size = found_size
            if candidate not in Q:
                Q.add(candidate)
                newly_added_q.append(candidate)

            used_grover = used_grover or new_used_grover

    for q in newly_added_q:
        Q.remove(q)
    return max_size, (total_expanded, total_grover), used_grover


if __name__ == "__main__":


    # correctness check
    for i in tqdm(range(4000)):
        graph = nx.erdos_renyi_graph(25, 0.5)

        # nx.draw(graph, with_labels=True)
        # plt.show()

        max_size, expanded = find_max_clique_rec(graph, set(), set(graph.nodes()), set(), 0)
        # print(f"Expanded: {expanded}")
        correct_size = len(get_max_clique(graph))
        if max_size != correct_size:
            print("WRONG")
            print(f"Found max size: {max_size}")
            print(f"Correct size: {correct_size}")
            print(f"seed: {i}")

