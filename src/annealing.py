from concurrent.futures import ProcessPoolExecutor, as_completed

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

def qa_max_clique_bqm(problem_instance, problem_size):
    graph = problem_instance["graph"]
    x = {i : Binary(i) for i in graph.nodes}
    complement_graph = nx.complement(graph)
    terms = [-x[i] for i in complement_graph.nodes]
    terms += [2*x[i]*x[j] for i,j in complement_graph.edges]
    bqm = sum(terms)
    return bqm

def qa_k_clique_bqm(problem_instance, problem_size):
    graph = problem_instance["graph"]
    k = problem_instance["k"]
    A = 1
    B = k

    x = {i: Binary(i) for i in graph.nodes}
    complement_graph = nx.complement(graph)

    size = sum(x[i] for i in graph.nodes)

    hamiltonian = A * (size - k)**2
    hamiltonian += B * sum(x[i] * x[j] for i, j in complement_graph.edges)

    return hamiltonian


def test_graph_qa(problem_instance, problem, validate_solutions, problem_size, iters=10, use_noisy_sampler=False):

    bqm = problem(problem_instance, problem_size)
    graph = problem_instance["graph"]
    sampler = PathIntegralAnnealingSampler() if use_noisy_sampler else SimulatedAnnealingSampler()

    sampleset = sampler.sample(bqm, num_reads=iters)

    node_order = list(graph.nodes)
    solutions = []
    for sample in sampleset.samples():
        bitstring = [sample[node] for node in node_order]
        solutions.append(bitstring)

    validation_results = validate_solutions(solutions)
    return validation_results

def qa_graph_worker(args):
    problem_instance, problem_size, problem, validate_solutions, iters, use_noise = args

    return test_graph_qa(
        problem_instance=problem_instance,
        problem=problem,
        validate_solutions=lambda bitstrings: validate_solutions(problem_instance, bitstrings, problem_size),
        problem_size=problem_size,
        iters=iters if iters is not None else problem_size["iters_per_graph"],
        use_noisy_sampler=use_noise
    )

def test_problem_sizes_qa(sizes, generate_instance, instance_count, problem, validate_solutions, iters=None, max_workers=4, use_noise=False):
    validation_results_per_size = []

    for s in sizes:
        print(f"--- Problem size: {s}")

        problem_instances = [generate_instance(s) for _ in range(instance_count)]

        args = [
            (problem_instance, s, problem, validate_solutions, iters, use_noise)
            for problem_instance in problem_instances
        ]

        validation_results_for_graphs = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(qa_graph_worker, arg)
                for arg in args
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                validation_results = future.result()
                validation_results_for_graphs.append(validation_results)

        validation_results_per_size.append(validation_results_for_graphs)

    return validation_results_per_size