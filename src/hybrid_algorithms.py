import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from scipy.special import comb
from tqdm import tqdm
from classical_algorithms import find_max_clique_rec
import json

def find_max_clique_hybrid(graph: nx.Graph, grover="naive", grover_condition=["first_level"], **kwargs):
    return find_max_clique_rec(graph, set(), set(graph.nodes()), set(), 0, grover=grover, grover_condition=grover_condition, **kwargs)

def benchmark_hybrid(filename, n_instances = 1000, n = 20, p = 0.92, grover="naive", grover_condition=["first_level"], **kwargs):
    better_count = 0
    used_grover_instances = 0
    k_classical = 1
    k_quantum = 1
    data_instances = []
    for i in tqdm(range(n_instances)):
        graph = nx.erdos_renyi_graph(n, p)
        result1 = find_max_clique_rec(graph, set(), set(graph.nodes()), set(), 0)
        result2 = find_max_clique_hybrid(graph, 
                                grover=grover, 
                                grover_condition=grover_condition,
                                **kwargs,
                                grover_max_n=n_instances)
        used_grover_instances += 1 if result2[2] else 0
        data_instances.append({
            "classical": result1,
            "quantum": result2
        })
        steps1 = k_classical * result1[1][0] + k_quantum * result1[1][1]
        steps2 = k_classical * result2[1][0] + k_quantum * result2[1][1]

        if steps1 > steps2:
            # print()
            # print(f"{i}: FOUND!")
            # print(steps1)
            # print(steps2)
            better_count += 1
    print(f"Better count: {better_count}")
    print(f"Grover instances: {used_grover_instances}")
    print(f"Percent better: {better_count / n_instances * 100}%")
    print(f"Percent better among used grover: {better_count / used_grover_instances * 100}%")

    with open(filename, "w") as f:
        json.dump({
            "meta": {
                "n_instances": n_instances,
                "n": n,
                "p": p,
                "used_grover_count": used_grover_instances
            },
            "data": data_instances
        }, f)

if __name__ == "__main__":
    pass

    graph = nx.erdos_renyi_graph(20, 0.95)
    # nx.draw(graph, with_labels=True)
    # plt.show()
    print(f"find_max_clique_rec: {find_max_clique_rec(graph, set(), set(graph.nodes()), set(), 0)}")
    print(f"find_max_clique_hybrid(naive): {find_max_clique_hybrid(graph, grover="naive")}")
    print(f"find_max_clique_hybrid(dicke): {find_max_clique_hybrid(graph, grover="dicke")}")
    print(f"find_max_clique_hybrid(dicke, min_density=0.8, max_density=0.98): {find_max_clique_hybrid(graph, 
                                                                                                      grover="dicke", 
                                                                                                      grover_condition=["density"], 
                                                                                                      min_density=0.9, 
                                                                                                      max_density=0.98,
                                                                                                      grover_max_n=20)}")

    # low p, direct grover
    # benchmark_hybrid("benchmarkResults/hybrid/dicke_n20_p06.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.6,
    #                 grover="dicke", 
    #                 )

    # high p, direct grover
    # benchmark_hybrid("benchmarkResults/hybrid/dicke_n20_p092.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.92,
    #                 grover="dicke", 
    #                 )
    
    # high p, density condition tests
    
    # benchmark_hybrid("benchmarkResults/hybrid/dicke_density92-98_n20_p092.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.92,
    #                 grover="dicke", 
    #                 grover_condition=["density"],
    #                 min_density=0.92, 
    #                 max_density=0.98
    #                 )
    benchmark_hybrid("benchmarkResults/hybrid/dicke_density85-98_n20_p092.json", 
                    n_instances=1000,
                    n=20,
                    p=0.92,
                    grover="dicke", 
                    grover_condition=["density"],
                    min_density=0.85, 
                    max_density=0.98
                    )
    benchmark_hybrid("benchmarkResults/hybrid/dicke_density80-98_n20_p092.json", 
                    n_instances=1000,
                    n=20,
                    p=0.92,
                    grover="dicke", 
                    grover_condition=["density"],
                    min_density=0.80, 
                    max_density=0.98
                    )


    # low p, density condition tests

    # benchmark_hybrid("benchmarkResults/hybrid/dicke_density90-98_n20_p06.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.6,
    #                 grover="dicke", 
    #                 grover_condition=["density"],
    #                 min_density=0.90, 
    #                 max_density=0.98
    #                 )
    # benchmark_hybrid("benchmarkResults/hybrid/dicke_density80-98_n20_p06.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.6,
    #                 grover="dicke", 
    #                 grover_condition=["density"],
    #                 min_density=0.80, 
    #                 max_density=0.98
    #                 )
    # benchmark_hybrid("benchmarkResults/hybrid/dicke_density70-98_n20_p06.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.6,
    #                 grover="dicke", 
    #                 grover_condition=["density"],
    #                 min_density=0.70, 
    #                 max_density=0.98
    #                 )


    # with min best size 12
    # benchmark_hybrid("benchmarkResults/hybrid/dicke_density85-98_minBestSize12_n20_p092.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.92,
    #                 grover="dicke", 
    #                 grover_condition=["density", "min_best_size"],
    #                 min_density=0.85, 
    #                 max_density=0.98,
    #                 min_best_size_num=5,
    #                 )

    # with min best size 5 in sparse graphs
    # benchmark_hybrid("benchmarkResults/hybrid/dicke_density70-98_minBestSize5_n20_p06.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.6,
    #                 grover="dicke", 
    #                 grover_condition=["density", "min_best_size"],
    #                 min_density=0.70, 
    #                 max_density=0.98,
    #                 min_best_size_num=5,
    #                 )
    # benchmark_hybrid("benchmarkResults/hybrid/dicke_density70-98_minBestSize4_n20_p06.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.6,
    #                 grover="dicke", 
    #                 grover_condition=["density", "min_best_size"],
    #                 min_density=0.70, 
    #                 max_density=0.98,
    #                 min_best_size_num=4,
    #                 )
    # benchmark_hybrid("benchmarkResults/hybrid/dicke_density70-98_minBestSize3_n20_p06.json", 
    #                 n_instances=1000,
    #                 n=20,
    #                 p=0.6,
    #                 grover="dicke", 
    #                 grover_condition=["density", "min_best_size"],
    #                 min_density=0.70, 
    #                 max_density=0.98,
    #                 min_best_size_num=3,
    #                 )