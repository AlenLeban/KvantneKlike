import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import json
from scipy.optimize import minimize

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from neal import SimulatedAnnealingSampler
from dimod import Binary, ExactSolver
from dwave.samplers import PathIntegralAnnealingSampler
from tqdm import tqdm

def qa_max_clique_bqm(graph, problem_size):
    x = {i : Binary(i) for i in graph.nodes}
    complement_graph = nx.complement(graph)
    terms = [-x[i] for i in complement_graph.nodes]
    terms += [2*x[i]*x[j] for i,j in complement_graph.edges]
    bqm = sum(terms)
    return bqm

def qa_k_clique_bqm(graph: nx.Graph, problem_size):
    k = problem_size["k"]
    A = k + 1
    B = 1

    x = {i: Binary(str(i)) for i in graph.nodes}
    complement_graph = nx.complement(graph)

    size = sum(x[i] for i in graph.nodes)

    hamiltonian = A * (size - k)**2
    hamiltonian += B * sum(x[i] * x[j] for i, j in complement_graph.edges)

    return hamiltonian


def test_graph_qa(graph, problem, validate_solutions, iters=10):

    # clique_percent_local = []
    # max_clique_percent_local = []
    solutions = []
    for i in range(iters):
        
        bqm = problem(graph)
        # sampler = SimulatedAnnealingSampler()
        sampler = PathIntegralAnnealingSampler()
        num_reads = 2
        sampleset = sampler.sample(bqm, num_reads=num_reads)
        # energies = np.array([s[1] for s in sampleset.record]) #  if is_clique(graph, s[0]) je do zdej blo vedno true
        # clique_sizes = [sum(r[0]) for r in sampleset.record] #  if is_clique(graph, s[0]) je do zdej blo vedno true
        # successful_runs = energies.shape[0]
        # clique_percent_local.append(successful_runs / num_reads)
        # unique, counts = np.unique(clique_sizes, return_counts=True)
        # exact_sampler = SimulatedAnnealingSampler()
        # exact_sampleset = exact_sampler.sample(bqm, num_reads=50)
        # print(exact_sampleset)
        solution = sampleset.lowest().record[0][0]
        solutions.append(solution)

        # exact_max_clique = sum(solution)
        # max_clique_percent_local.append(np.sum(np.array(clique_sizes) == np.full_like(clique_sizes, exact_max_clique)) / num_reads)
        # print(f"Exact solution: {exact_max_clique}")
        # max_cliques_found.append()
    validation_results = validate_solutions(solutions)
    return validation_results

def test_problem_sizes_qa(sizes, generate_instance, instance_count, problem, validate_solutions, iters=None):
    validation_results_per_size = []
    for s in sizes:
        print(f"--- Problem size: {s}")
        graphs = [generate_instance(s) for i in range(instance_count)]
        validation_results_for_graphs = []
        for graph in tqdm(graphs):
            validation_results = test_graph_qa(
                graph, 
                problem=lambda g: problem(g, s),
                validate_solutions=lambda x: validate_solutions(graph, x, s),
                iters=iters if iters != None else s["iters_per_graph"]
            )
            
            validation_results_for_graphs.append(validation_results)
            
        validation_results_per_size.append(validation_results_for_graphs)
        
    return validation_results_per_size