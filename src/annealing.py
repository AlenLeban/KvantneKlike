from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix, diags, dok_matrix, identity, kron
from scipy.sparse.linalg import eigsh
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

from utils import generate_graph_with_k_clique, generate_random_graph_instance, is_number

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

def build_initial_hamiltonian_sparse(n):
    sx = csr_matrix([[0, 1],
                     [1, 0]], dtype=float)
    I = identity(2, format="csr", dtype=float)

    H = csr_matrix((2**n, 2**n), dtype=float)

    for i in range(n):
        term = None
        for j in range(n):
            op = sx if i == j else I
            term = op if term is None else kron(term, op, format="csr")
        H -= term

    return H

def build_problem_hamiltonian_sparse(bqm, n):
    energies = np.empty(2**n)

    for state in range(2**n):
        sample = {i: (state >> i) & 1 for i in range(n)}
        energies[state] = bqm.energy(sample)

    return diags(energies, format="csr")


if __name__ == "__main__":
    pass
    # n = 16
    # k = 7

    # problem_instance = generate_random_graph_instance({"n": n, "p": 0.5})
    # bqm = qa_max_clique_bqm(problem_instance, {"k": None})
    # # matrix_elements = bqm.to_qubo()
    # # dim = max(matrix_elements[0].keys(), key=lambda x: max(x[0], x[1]))
    # # max_dim = max(dim[0], dim[1])
    # # sparse_matrix = dok_matrix((max_dim+1, max_dim+1))
    # # for pos in matrix_elements[0]:
    # #     sparse_matrix[pos[0], pos[1]] = matrix_elements[0][pos]

    # # Q = sparse_matrix.toarray()
    # # Q = (Q + Q.T) / 2

    # print("Preparing hamiltonians")
    # H_initial = build_initial_hamiltonian_sparse(problem_instance["graph"].number_of_nodes())
    # H_problem = build_problem_hamiltonian_sparse(bqm, n)

    # prev_vec = None
    # steps = 50
    # first_k = 8
    # eigs_array = []
    
    # sampler = SimulatedAnnealingSampler()

    # sampleset = sampler.sample(bqm, num_reads=10)
    # print(sampleset)

    # max_degeneracy = 0
    # H_initial_min = 999
    # H_problem_diagonal = np.sort(H_problem.diagonal())
    # H_problem_range = H_problem_diagonal[-1] - H_problem_diagonal[0]

    # print("Simulating")
    # for step in tqdm(range(steps-1)):
    #     s = step / steps
    #     Hs = (1 - s) * H_initial + s * H_problem

    #     if step == steps-2:
    #         H_problem_diag = np.sort(H_problem.diagonal())
    #         vals = H_problem_diag[:first_k]

    #     else:
    #         vals, vecs = eigsh(
    #             Hs,
    #             k=first_k,
    #             which="SA",
    #             v0=prev_vec,
    #             return_eigenvectors=True,
    #             tol=1e-4,
    #         )

    #     order = np.argsort(vals)
    #     vals = vals[order]
    #     vecs = vecs[:, order]
    #     vals -= vals[0]
    #     prev_vec = vecs[:, 0]
    #     eigs_array.append(vals)

    # eigs_matrix = np.array(eigs_array)

    # ground_tol = 1e-7
    # ground = 0
    # ground_degeneracy = np.sum(abs(eigs_array[-1] - ground) < ground_tol)
    # print(f"Ground state degeneracy: {ground_degeneracy}")
    # gaps_array = np.min(eigs_matrix[:,ground_degeneracy:], axis=1)
    # min_gap_index = np.argmin(gaps_array)
    # min_gap = gaps_array[min_gap_index]
    # print(f"Minimum gap: {min_gap}")
    
    # # print(H_problem_diag[:first_k])
    # # eigs_array.append(H_problem_diag[:first_k])
    # # print(eigs_array)
    

    # _, ax = plt.subplots(1, 2)
    # ax[0].set_title("Energy levels throughout annealing (relative\n to ground state)")
    # ax[0].set_ylabel("Energy")
    # for i in range(first_k):
    #     ax[0].plot(np.arange(steps-1)/(steps-1), eigs_matrix[:,i])
    # # plt.plot(np.arange(len(gaps_array)) / steps, gaps_array, label = "Gap")
    # ax[0].vlines([min_gap_index / (steps-1)], 0, min_gap, color="red")
    # ax[0].set_xlabel("s")
    # ax[1].plot(np.arange(len(gaps_array))/(steps-1), gaps_array)
    # ax[0].grid()
    # ax[1].grid()
    # ax[1].set_ylim(bottom=0)
    # plt.show()
    # plt.figure()
    # nx.draw(problem_instance["graph"], with_labels=True)
    # plt.show()
