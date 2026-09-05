from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy as sc
import gc
import time
from scipy.sparse import csr_matrix, diags, dok_matrix, identity, kron
from scipy.sparse.linalg import eigsh
from matplotlib import pyplot as plt
import networkx as nx
import json
from scipy.optimize import minimize
import qutip as qt

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


def test_graph_qa(problem_instance, problem, validate_solutions, problem_size, iters=10, use_noisy_sampler=False, return_eigenenergies=False):

    bqm = problem(problem_instance, problem_size)
    graph = problem_instance["graph"]
    solutions = []
    nodes = sorted(graph.nodes)
    n_qubits = len(nodes)
    annealing_time = problem_size.get("T", 20)
    n_steps = problem_size.get("steps", 2000)
    node_to_qubit = {
        node: qubit
        for qubit, node in enumerate(nodes)
    }

    H_problem = bqm_to_qutip_hamiltonian(
        bqm,
        node_to_qubit
    )

    energies = None

    times, final_state, energies = run_quantum_annealing(
        H_problem,
        n_qubits,
        annealing_time=annealing_time,
        n_steps=n_steps,
        return_eigenenergies=return_eigenenergies
    )

    probs = np.abs(final_state.full().flatten())**2

    success_probability = 0.0

    for state in range(2**n_qubits):
        solution = [
            1 if state & 2**(n_qubits - 1 - i) else 0
            for i in range(n_qubits)
        ]

        if validate_solutions([solution])["found_solution"]:
            success_probability += probs[state]
    # max_amplitude_state = np.argmax(probs, axis=0)
    # print(final_state)
    # print("{0:b}".format(max_amplitude_state[0]))
    # print(max_amplitude_state)
    # solution = [1 if max_amplitude_state & 2**(n_qubits - 1 - i) > 0 else 0 for i in range(n_qubits)]
    # print(solution)
    # solutions = []

    # sampler = PathIntegralAnnealingSampler() if use_noisy_sampler else SimulatedAnnealingSampler()
    
    # sampleset = sampler.sample(bqm, num_reads=iters)
    # print(solutions)
    # node_order = list(graph.nodes)
    # solutions = []
    # for sample in sampleset.samples():
    #     bitstring = [sample[node] for node in node_order]
    #     solutions.append(bitstring)

    # validation_results = validate_solutions(solutions)
    return_payload = [
        {
            "success_probability": success_probability
        }
    ]
    if return_eigenenergies:
        return_payload.append(energies)
    return return_payload[0] if len(return_payload) == 1 else return_payload



def qa_graph_worker(args):
    problem_instance, problem_size, problem, validate_solutions, iters, use_noise, return_eigenenergies = args

    return test_graph_qa(
        problem_instance=problem_instance,
        problem=problem,
        validate_solutions=lambda bitstrings: validate_solutions(problem_instance, bitstrings, problem_size),
        problem_size=problem_size,
        iters=iters if iters is not None else problem_size["iters_per_graph"],
        use_noisy_sampler=use_noise,
        return_eigenenergies=return_eigenenergies
    )

def test_problem_sizes_qa(sizes, generate_instance, instance_count, problem, validate_solutions, iters=None, max_workers=4, use_noise=False, return_eigenenergies=False):
    validation_results_per_size = []

    for s in sizes:
        print(f"--- Problem size: {s}")

        problem_instances = [generate_instance(s) for _ in range(instance_count)]

        args = [
            (problem_instance, s, problem, validate_solutions, iters, use_noise, return_eigenenergies)
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



def bqm_to_qutip_hamiltonian(bqm, node_to_qubit):
    """
    Convert a dimod BinaryQuadraticModel into a QuTiP
    Ising Hamiltonian.

    Uses:

        x_i = (1 - Z_i) / 2
    """

    n_qubits = len(node_to_qubit)

    identity = qt.tensor(
        [qt.qeye(2)] * n_qubits
    )

    H = bqm.offset * identity

    # Linear terms
    for node, h in bqm.linear.items():

        q = node_to_qubit[node]

        Z = pauli_z(n_qubits, q)

        H += h / 2 * identity
        H -= h / 2 * Z

    # Quadratic terms
    for (u, v), J in bqm.quadratic.items():

        q_u = node_to_qubit[u]
        q_v = node_to_qubit[v]

        Z_u = pauli_z(n_qubits, q_u)
        Z_v = pauli_z(n_qubits, q_v)

        H += J / 4 * identity
        H -= J / 4 * Z_u
        H -= J / 4 * Z_v
        H += J / 4 * Z_u * Z_v

    return H

def pauli_z(n_qubits, qubit):
    operators = []

    for i in range(n_qubits):
        operators.append(
            qt.sigmaz() if i == qubit else qt.qeye(2)
        )

    return qt.tensor(operators)


def pauli_x(n_qubits, qubit):
    operators = []

    for i in range(n_qubits):
        operators.append(
            qt.sigmax() if i == qubit else qt.qeye(2)
        )

    return qt.tensor(operators)

def transverse_field_hamiltonian(n_qubits):

    H = 0

    for q in range(n_qubits):
        H -= pauli_x(n_qubits, q)

    return H

def initial_plus_state(n_qubits):

    plus = (
        qt.basis(2, 0)
        + qt.basis(2, 1)
    ).unit()

    return qt.tensor(
        [plus] * n_qubits
    )

def run_quantum_annealing(
    H_problem,
    n_qubits,
    annealing_time=10.0,
    n_steps=1000,
    return_eigenenergies = False
):

    H_initial = transverse_field_hamiltonian(
        n_qubits
    )

    psi0 = initial_plus_state(n_qubits)

    times = np.linspace(
        0,
        annealing_time,
        n_steps
    )

    def H_t(t, **kwargs):

        s = t / annealing_time

        return (
            (1.0 - s) * H_initial
            + s * H_problem
        )


    result = qt.sesolve(
        H_t,
        psi0,
        times,
        options={"store_final_state": True, "store_states": False}
    )

    final_state = result.final_state
    energies_array = None
    if return_eigenenergies:
        energies_array = []
        for t in tqdm(times[:-1]):
            H_at_t = H_t(t)
            energies = np.real(H_at_t.eigenenergies(sparse=True, eigvals=5, sort="low"))
            energies_array.append(energies.tolist()) 
    gc.collect()
    del H_problem
    del psi0
    del H_initial
    del result
    return times, final_state, energies_array


if __name__ == "__main__":

    # problem_instance = {
    #     "graph": nx.erdos_renyi_graph(13, 0.6),
    #     "k": 5
    # }

    # plt.figure()
    # nx.draw(problem_instance["graph"], with_labels=True)
    # plt.show()

    # nodes = sorted(problem_instance["graph"].nodes)
    # n_qubits = len(nodes)
    # node_to_qubit = {
    #     node: qubit
    #     for qubit, node in enumerate(nodes)
    # }

    # bqm_expression = qa_k_clique_bqm(
    #     problem_instance,
    #     {"k": 3, "B": 1.1}
    # )


    # bqm = bqm_expression

    # H_problem = bqm_to_qutip_hamiltonian(
    #     bqm,
    #     node_to_qubit
    # )
    # # print(H_problem)

    # times, result = run_quantum_annealing(
    #     H_problem,
    #     n_qubits,
    #     annealing_time=20.0,
    #     n_steps=2000
    # )

    # final_state = result.states[-1]
    # probs = np.abs(final_state.full())**2
    # max_amplitude_state = np.argmax(probs, axis=0)
    # # print(final_state)
    # print("{0:b}".format(max_amplitude_state[0]))

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
