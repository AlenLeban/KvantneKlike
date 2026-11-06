import numpy as np
import networkx as nx
from max_clique_github.max_clique.main import get_max_clique
import timeit
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import math
from utils import er_max_clique_size

def generate_k_optimal_instance(n, p, k, seed=None):
    while True:
        graph = nx.erdos_renyi_graph(n, p, seed=seed)
        mc = get_max_clique(graph)
        if len(mc) == k:
            return graph



def generate_block_diagonal_hard_instance(n, p, k_max, seed=None):
    graph = nx.erdos_renyi_graph(n, p)
    edges = []
    edges.extend([(i,j) for i in range(k_max) for j in range(k_max) if i < j])
    # print(edges)
    for b in range(k_max, n, k_max-1):
        new_edges = [(i,j) for i in range(b, min(b + k_max - 1, n)) for j in range(b, min(b + k_max - 1, n)) if i < j]
        # print(new_edges)
        edges.extend(new_edges)

    # print(edges)
    graph.add_edges_from(edges)
    return graph

if __name__ == "__main__":

    # nx.draw(graph, with_labels=True)
    # plt.show()

    def test(seed):
        graph = nx.erdos_renyi_graph(10, 0.5, seed=seed)
        return graph

    def test_hard(seed):
        graph = generate_block_diagonal_hard_instance(10, 0.5, 5, seed=seed)
        return graph

    def test_k_optimal(seed):
        graph = generate_k_optimal_instance(10, 0.5, 4, seed=seed)
        return graph

    def benchmark(generate_function, fig=None, **kwargs):
        measurements = []
        data = []
        samples = 10
        for r in tqdm(range(50000)):
            graph = generate_function(None)
            t1 = timeit.default_timer()
            for s in range(samples):
                mc = get_max_clique(graph)
            t2 = timeit.default_timer()
            data.append((mc, graph))
            measurements.append(t2 - t1)

        zipped = zip(data, measurements)
        q = 0.9
        mean_measurement = np.mean(measurements)
        std_measurement = np.std(measurements)
        print(mean_measurement)
        print(std_measurement)
        measurements = np.array(measurements)
        print(len(measurements))
        q_low = np.quantile(measurements, 1-q)
        q_high = np.quantile(measurements, q)
        print(len(measurements))
        easy_instances = list(filter(lambda x: x[1] < q_low and x[1] < std_measurement, zipped))
        zipped = zip(data, measurements)
        hard_instances = list(filter(lambda x: x[1] > q_high and x[1] < std_measurement, zipped))
        print(len(hard_instances))
        if fig == None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig)
        measurements = measurements[np.abs(measurements - mean_measurement) < std_measurement]
        meas_range = max(measurements) - min(measurements)
        bins = math.floor(meas_range / 50 * 3000000)
        print(bins)
        plt.hist(measurements, bins=bins, color=kwargs.get("color", None), alpha=0.5)
        plt.axvline(mean_measurement, color=kwargs.get("color", None))
        plt.axvline(q_low, color=kwargs.get("color", None))
        plt.axvline(q_high, color=kwargs.get("color", None))
        # plt.show()

        return fig, hard_instances

    fig, _ = benchmark(test, color="tab:blue")
    fig, hard_instances = benchmark(test_hard, fig=fig, color="tab:red")
    # fig, _ = benchmark(test_k_optimal, fig=fig, color="tab:green")
    plt.show()

    with open("benchmark_data/er_hard_classical/max_clique/er_10_05", "w") as f:
        jsons = []
        for instance in hard_instances:
            max_clique_size = len(instance[0][0])
            graph:nx.Graph = instance[0][1]
            jsons.append({
                "graph": list(nx.generate_adjlist(graph)),
                "k": max_clique_size
            })
        json.dump(jsons, f)

    # print(easy_instances[0][0])
    # example_easy_graph = easy_instances[0][0][1]
    # nx.draw(example_easy_graph, with_labels=True)
    # plt.show()

    # print(hard_instances[0][0])
    # example_hard_graph = hard_instances[0][0][1]
    # nx.draw(example_hard_graph, with_labels=True)
    # plt.show()
