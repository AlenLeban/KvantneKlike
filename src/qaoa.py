from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from matplotlib import pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, Batch, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.circuit.library import QAOAAnsatz
import networkx as nx
import json
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorEstimator
from multiprocessing import Pool
from tqdm import tqdm
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from utils import to_bitstring


def build_maxcut_paulis(graph: nx.Graph):
    pauli_list = []
    
    for e in graph.edges():
        pauli_list.append(("ZZ", [e[0], e[1]], 1))
    return pauli_list

def build_maxclique_mis_paulis(graph: nx.Graph, problem_size):
    complement = nx.complement(graph)
    # solve max independent set
    pauli_list = []
    A = 1
    for n in complement.nodes():
        pauli_list.append((("Z"), [n], A))

    B = 2/4
    for e in complement.edges():
        pauli_list.append(("II", [e[0], e[1]], B))
        pauli_list.append(("IZ", [e[0], e[1]], -B))
        pauli_list.append(("ZI", [e[0], e[1]], -B))
        pauli_list.append(("ZZ", [e[0], e[1]], B))
        
    return pauli_list

def build_kclique_paulis(graph, k):
    
    pauli_list = []
    B = 1
    A = B*k + 1
    observable_dict = dict()
    N = graph.number_of_edges()
    M = graph.number_of_edges()

    for n in graph.nodes():
        observable_dict[("Z", (n,))] = observable_dict.get(("Z", (n,)), 0) + A*k - A*0.5*N
        for m in graph.nodes():
            if n >= m:
                continue
            observable_dict[("ZZ", (n,m))] = observable_dict.get(("ZZ", (n,m)), 0) + A*0.5

    for e in graph.edges():
        observable_dict[("Z", (e[1],))] = observable_dict.get(("Z", (e[1],)), 0) + B/4
        observable_dict[("Z", (e[0],))] = observable_dict.get(("Z", (e[0],)), 0) + B/4
        sorted_nm = sorted([e[0], e[1]])
        observable_dict[("ZZ", (sorted_nm[0],sorted_nm[1]))] = observable_dict.get(("ZZ", (sorted_nm[0], sorted_nm[1])), 0) - B/4


    for (pauli, qubits), coeff in observable_dict.items():
        pauli_list.append((pauli, list(qubits), coeff))
    return pauli_list

def build_kclique_paulis_mis(graph, problem_size):
    
    complement = nx.complement(graph)
    pauli_list = []
    k = problem_size["k"]
    B = 1
    A = B*k + 1
    observable_dict = dict()
    N = graph.number_of_nodes()
    M = graph.number_of_edges()

    for n in graph.nodes():
        observable_dict[("Z", (n,))] = observable_dict.get(("Z", (n,)), 0) + A*k - A*0.5*N
        for m in graph.nodes():
            if n >= m:
                continue
            observable_dict[("ZZ", (n,m))] = observable_dict.get(("ZZ", (n,m)), 0) + A*0.5

    for e in complement.edges():
        observable_dict[("Z", (e[1],))] = observable_dict.get(("Z", (e[1],)), 0) - B/4
        observable_dict[("Z", (e[0],))] = observable_dict.get(("Z", (e[0],)), 0) - B/4
        sorted_nm = sorted([e[0], e[1]])
        observable_dict[("ZZ", (sorted_nm[0],sorted_nm[1]))] = observable_dict.get(("ZZ", (sorted_nm[0], sorted_nm[1])), 0) + B/4


    for (pauli, qubits), coeff in observable_dict.items():
        pauli_list.append((pauli, list(qubits), coeff))

    return pauli_list

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    result = job.result()[0]
    cost = result.data.evs

    return cost

def test_graph_qaoa(graph, problem, validate_solutions, iters=5, num_layers=2, use_noisy_optimizer=False, use_noisy_sampling=False):

    validation_results = []
    # exc = ThreadPoolExecutor(max_workers=2)
    if use_noisy_optimizer:
        backend = AerSimulator.from_backend(FakeBrisbane())
        estimator = Estimator(mode=backend)
    else:
        backend = AerSimulator()
        estimator = StatevectorEstimator()
    backend.set_options(statevector_parallel_threshold=100)
    # backend.set_options(executor=exc)
    # backend.set_options(
    #     max_parallel_threads = 0,
    #     max_parallel_experiments = 0,
    #     max_parallel_shots = 1,
    #     statevector_parallel_threshold = 16
    # )
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 1000
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    paulis = problem(graph)
    cost_hamiltonian = SparsePauliOp.from_sparse_list(paulis, graph.number_of_nodes())

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=num_layers)
    circuit.measure_all()
    candidate_circuit = pm.run(circuit)
    candidate_circuit_no_meas = candidate_circuit.remove_final_measurements(inplace=False)
    found_bitstrings = []
    cumulative_optimized_circuit_depth = 0
    avg_number_evaluations = 0
    optimized_circuits = []
    for i in range(iters):


        initial_params = np.random.rand(2*num_layers) * 2 * np.pi


        result = minimize(
            cost_func_estimator,
            initial_params,
            args=(candidate_circuit_no_meas, cost_hamiltonian, estimator),
            method="COBYLA",
            # options={"maxiter": 200},
            tol=1e-2
        )
        # print(f"Energy: {cost_func_estimator(result.x, candidate_circuit_no_meas, cost_hamiltonian, estimator)}")
        optimized_circuit = candidate_circuit.assign_parameters(result.x)
        cumulative_optimized_circuit_depth += optimized_circuit.depth()
        avg_number_evaluations += result.nfev
        optimized_circuits.append(optimized_circuit)

    # pub = (optimized_circuit, )

    job = sampler.run(optimized_circuits, shots=1000)
    for res in job.result():
        counts_int = res.data.meas.get_int_counts()

        most_likely = max(counts_int, key=counts_int.get)

        most_likely_bitstring = to_bitstring(most_likely, graph.number_of_nodes())
        most_likely_bitstring.reverse()

        # print("Result bitstring:", most_likely_bitstring)
        found_bitstrings.append(most_likely_bitstring)
    
    
    
    validation_results = validate_solutions(found_bitstrings)
    # print(f"Valid cliques: {valid_cliques}/{iters}")
    # print(f"Valid k-cliques: {k_cliques}/{iters}")

    average_opimized_circuit_depth = cumulative_optimized_circuit_depth / iters
    avg_number_evaluations /= iters
    return validation_results, average_opimized_circuit_depth, avg_number_evaluations

def _qaoa_graph_worker(args):
    graph, s, problem, validate_solutions, layers, iters, use_noisy_optimizer = args

    return test_graph_qaoa(
        graph,
        problem=lambda g: problem(g, s),
        validate_solutions=lambda bitstrings: validate_solutions(graph, bitstrings, s),
        iters=iters if iters is not None else s["iters_per_graph"],
        num_layers=layers if layers is not None else s["layers"],
        use_noisy_optimizer=use_noisy_optimizer
    )

def test_problem_sizes_qaoa(
    sizes,
    generate_instance,
    instance_count,
    problem,
    validate_solutions,
    layers=4,
    iters=None,
    max_workers=None,
    use_noisy_optimizer=False
):
    validation_results_per_size = []
    depths_per_size = []
    evaluations_per_size = []

    for s in sizes:
        print(f"--- Problem size: {s}")

        graphs = [generate_instance(s) for _ in range(instance_count)]

        worker_args = [
            (graph, s, problem, validate_solutions, layers, iters, use_noisy_optimizer)
            for graph in graphs
        ]

        validation_results_for_graphs = []
        depths_per_graph = []
        evaluations_per_graph = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_qaoa_graph_worker, args)
                for args in worker_args
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                validation_results, optimized_circuit_depth, avg_num_evaluations = future.result()

                validation_results_for_graphs.append(validation_results)
                depths_per_graph.append(optimized_circuit_depth)
                evaluations_per_graph.append(avg_num_evaluations)

        validation_results_per_size.append(validation_results_for_graphs)
        depths_per_size.append(depths_per_graph)
        evaluations_per_size.append(evaluations_per_graph)

    return validation_results_per_size, depths_per_size, evaluations_per_size