from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy as sc
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigs
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

from utils import generate_graph_with_k_clique, is_number

def qa_max_clique_bqm(problem_instance, problem_size):
    graph = problem_instance["graph"]
    x = {i : Binary(i) for i in graph.nodes}
    complement_graph = nx.complement(graph)
    terms = [-x[i] for i in complement_graph.nodes]
    B = 1.1 if "B" not in problem_size else problem_size["B"]
    terms += [B*x[i]*x[j] for i,j in complement_graph.edges]
    bqm = sum(terms)
    return bqm

def qa_k_clique_bqm(problem_instance, problem_size):
    graph = problem_instance["graph"]
    k = problem_instance["k"]
    A = 1
    if "B" in problem_size:
        if problem_size["B"] == "k":
            B = k
        elif is_number(problem_size["B"]):
            B = float(problem_size["B"])
    else:
        B = 1
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

def build_initial_hamiltonian(n):
    sx = np.array([[0, 1],
                   [1, 0]], dtype=float)
    I = np.eye(2)

    H_init = np.zeros((2**n, 2**n))

    for i in range(n):
        ops = [I] * n
        ops[i] = sx

        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)

        H_init -= term

    return H_init


if __name__ == "__main__":

    n = 12
    k = 6

    problem_instance = generate_graph_with_k_clique(n, 0.5, k)
    bqm = qa_k_clique_bqm(problem_instance, {"k": k})
    matrix_elements = bqm.to_qubo()
    dim = max(matrix_elements[0].keys(), key=lambda x: max(x[0], x[1]))
    max_dim = max(dim[0], dim[1])
    sparse_matrix = dok_matrix((max_dim+1, max_dim+1))
    for pos in matrix_elements[0]:
        sparse_matrix[pos[0], pos[1]] = matrix_elements[0][pos]

    Q = sparse_matrix.toarray()
    Q = (Q + Q.T) / 2

    H_initial = build_initial_hamiltonian(problem_instance["graph"].number_of_nodes())
    H_problem = np.zeros_like(H_initial)


    for state in range(H_initial.shape[0]):
        sample = {i: (state >> i) & 1 for i in range(n)}
        H_problem[state, state] = bqm.energy(sample)

    H_initial /= 2
    print(H_problem)
    print(H_problem.shape)
    print(H_initial.shape)
    initial_eigs = np.linalg.eigvalsh(H_initial)
    # print(H_initial)
    problem_eigs = np.linalg.eigvalsh(H_problem)
    initial_range = max(initial_eigs) - min(initial_eigs)
    problem_range = max(problem_eigs) - min(problem_eigs)
    initial_center = min(initial_eigs)
    problem_center = min(problem_eigs)


    plt.figure()
    nx.draw(problem_instance["graph"], with_labels=True)
    plt.show()

    eigs_array = []
    first_k = 10
    steps = 10
    for step in tqdm(range(steps)):
        s = step / steps
        Hs = (1 - s) * H_initial + s * H_problem
        offset = (1 - s) * initial_center + s * problem_center
        scales = (1 - s) * initial_range + s * problem_range
        eigenvalues = (np.linalg.eigvalsh(Hs)[:first_k] - offset) / scales
        eigs_array.append(eigenvalues)

    eigs_matrix = np.array(eigs_array)
    plt.figure()
    for k in range(first_k):
        plt.plot(np.arange(steps)/steps, eigs_matrix[:,k])
    plt.show()
