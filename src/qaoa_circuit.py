
import re

import numpy as np
from qiskit import QuantumCircuit, generate_preset_pass_manager
import random
from benchmark import helper_validate_max_clique_solutions
from qaoa import build_maxclique_mis_paulis, quick_run_qaoa, test_graph_qaoa
from utils import generate_random_graph_instance, validate_max_clique_solutions
import networkx as nx
from matplotlib import pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator

def myQAOAAnsatzDFS(cost_hamiltonian, reps, problem_instance=None):
    
    qc = QuantumCircuit(cost_hamiltonian.num_qubits)
    qc.h(qc.qubits)

    betas = [Parameter(f"beta_{i}") for i in range(reps)]
    gammas = [Parameter(f"gamma_{i}") for i in range(reps)]
    edge_coloring = None
    graph_dfs_tree = None
    dfs_edges = []

    # find DFS tree first, dfs edges can be optimized sequentially
    graph: nx.Graph = problem_instance["graph"]
    complement_graph = nx.complement(graph)
    graph_dfs_tree = nx.dfs_tree(complement_graph)
    dfs_edges = list(graph_dfs_tree.edges)
    # plt.figure()
    # print(dfs_edges)
    # nx.draw(graph_dfs_tree, with_labels=True)
    # plt.show()

    complement_graph.remove_edges_from(dfs_edges)
    # then perform coloring on the remaining edges
    edge_graph = nx.line_graph(complement_graph)
    edge_coloring = nx.greedy_color(edge_graph, strategy="independent_set")
    print(f"Coloring: {edge_coloring}")

    qubit_coeffs = dict()
    pair_coeffs = dict()
    pair_groups = dict()
    for paulis, coeff in zip(cost_hamiltonian.paulis, cost_hamiltonian.coeffs):
        num_z = len(str(paulis).replace("I", ""))
        if num_z == 1:
            qubit_index = qc.num_qubits - str(paulis).index("Z") - 1
            qubit_coeffs[qubit_index] = qubit_coeffs.get(qubit_index, 0) + coeff
        elif num_z == 2:
            positions = sorted([qc.num_qubits - match.start() - 1 for match in re.finditer("Z", str(paulis))])
            pair_coeffs[tuple(positions)] = pair_coeffs.get(tuple(positions), 0) + coeff
            if tuple(positions) not in edge_coloring:
                continue
            edge_color = edge_coloring[tuple(positions)]
            if edge_color not in pair_groups:
                pair_groups[edge_color] = []
            pair_groups[edge_color].append(tuple(positions))

    # print(pair_coeffs)

    

    used_qubits = set()
    for r in range(reps):

        # problem hamiltonian
        if r == 0:
            pass

        for color, pairs in sorted(pair_groups.items(), reverse=True, key=lambda x: len(x[1])):
            for pair in pairs:
                if r == 0 and not complement_graph.has_edge(pair[0], pair[1]):
                    continue
                c = pair_coeffs[pair]
                positions = pair
                # if positions[0] in used_qubits or r > 0:
                qc.cx(positions[1], positions[0])
                qc.rz(2 * gammas[r] * c, positions[0])
                qc.cx(positions[1], positions[0])
                used_qubits.add(positions[0])
                used_qubits.add(positions[1])

        for q,c in qubit_coeffs.items():
            qc.rz(2 * gammas[r] * c, q)
        # qc.barrier()

        # mixer hamiltonian
        qc.rx(2 * betas[r], qc.qubits)

    return qc

def myQAOAAnsatz(cost_hamiltonian, reps, use_coloring=False, problem_instance=None, optimize_cnots=False, use_dfs=False):

    qc = QuantumCircuit(cost_hamiltonian.num_qubits)
    qc.h(qc.qubits)

    betas = [Parameter(f"beta_{i}") for i in range(reps)]
    gammas = [Parameter(f"gamma_{i}") for i in range(reps)]
    edge_coloring = None
    if use_coloring:
        graph = problem_instance["graph"]
        complement_graph = nx.complement(graph)
        edge_graph = nx.line_graph(complement_graph)
        edge_coloring = nx.greedy_color(edge_graph, strategy="independent_set")
        # print(f"Coloring: {edge_coloring}")

    graph_dfs_tree = None
    dfs_edges = []

    qubit_coeffs = dict()
    pair_coeffs = dict()
    pair_groups = dict()
    for paulis, coeff in zip(cost_hamiltonian.paulis, cost_hamiltonian.coeffs):
        num_z = len(str(paulis).replace("I", ""))
        if num_z == 1:
            qubit_index = qc.num_qubits - str(paulis).index("Z") - 1
            qubit_coeffs[qubit_index] = qubit_coeffs.get(qubit_index, 0) + coeff
        elif num_z == 2:
            positions = sorted([qc.num_qubits - match.start() - 1 for match in re.finditer("Z", str(paulis))])
            pair_coeffs[tuple(positions)] = pair_coeffs.get(tuple(positions), 0) + coeff
            if use_coloring:
                edge_color = edge_coloring[tuple(positions)]
                if edge_color not in pair_groups:
                    pair_groups[edge_color] = []
                pair_groups[edge_color].append(tuple(positions))

    # print(pair_groups)

    used_qubits = set()
    no_dfs_pair_groups = dict()
    if use_dfs:
        graph: nx.Graph = problem_instance["graph"]
        complement_graph = nx.complement(graph)
        graph_dfs_tree = nx.dfs_tree(complement_graph)
        dfs_edges = nx.dfs_edges(complement_graph)
        # plt.figure()
        # print(dfs_edges)
        # nx.draw(graph_dfs_tree, with_labels=True)
        # plt.show()
        # qc.barrier()
        dfs_edges_list = list(dfs_edges)
        # print(pair_coeffs)
        for e in dfs_edges_list:
            positions = tuple(e)
            c = pair_coeffs[tuple(sorted(positions))]
            qc.rz(2 * gammas[0] * c, positions[1])
            # print(2 * gammas[0] * c)
            
        for e in dfs_edges_list:
            # print(e)
            # print(pair_coeffs)
            positions = tuple(e)
            qc.cx(positions[0], positions[1])
            used_qubits.add(positions[0])
            used_qubits.add(positions[1])
        # qc.barrier()

        no_dfs_graph = complement_graph.copy()
        no_dfs_graph.remove_edges_from(dfs_edges_list)
        no_dfs_edge_graph = nx.line_graph(no_dfs_graph)
        no_dfs_edge_coloring = nx.greedy_color(no_dfs_edge_graph, strategy="independent_set")
        # print(f"no dfs Coloring: {no_dfs_edge_coloring}")

        for e,edge_color in no_dfs_edge_coloring.items():
            if edge_color not in no_dfs_pair_groups:
                no_dfs_pair_groups[edge_color] = []
            no_dfs_pair_groups[edge_color].append(e)

        for color, pairs in sorted(no_dfs_pair_groups.items(), reverse=True, key=lambda x: len(x[1])):
            for pair in pairs:
                c = pair_coeffs[pair]
                positions = pair
                qc.cx(positions[1], positions[0])
                qc.rz(2 * gammas[0] * c, positions[0])
                qc.cx(positions[1], positions[0])
                used_qubits.add(positions[0])
                used_qubits.add(positions[1])

        for q,c in qubit_coeffs.items():
            qc.rz(2 * gammas[0] * c, q)
        # qc.barrier()
        # mixer hamiltonian
        qc.rx(2 * betas[0], qc.qubits)

    # qc.barrier()

    # print(no_dfs_pair_groups)
    for r in range(1, reps) if use_dfs else range(reps):

        # problem hamiltonian
        # if r == 0 and use_dfs:
        #     for e in dfs_edges:


        if use_coloring:
            for color, pairs in sorted(pair_groups.items(), reverse=True, key=lambda x: len(x[1])):
                for pair in pairs:
                    c = pair_coeffs[pair]
                    positions = pair
                    if positions[0] in used_qubits or not optimize_cnots or r > 0:
                        qc.cx(positions[1], positions[0])
                    qc.rz(2 * gammas[r] * c, positions[0])
                    qc.cx(positions[1], positions[0])
                    used_qubits.add(positions[0])
                    used_qubits.add(positions[1])
        else:
            for q,c in pair_coeffs.items():
                positions = q
                qc.cx(positions[1], positions[0])
                qc.rz(2 * gammas[r] * c, positions[0])
                qc.cx(positions[1], positions[0])

        for q,c in qubit_coeffs.items():
            qc.rz(2 * gammas[r] * c, q)
        # qc.barrier()

        # mixer hamiltonian
        qc.rx(2 * betas[r], qc.qubits)

    return qc

if __name__ == "__main__":

    # from qiskit.quantum_info import SparsePauliOp
    # from qiskit.circuit.library import qaoa_ansatz

    # cost_operator = SparsePauliOp(["ZZII", "IIZI", "ZIIZ"])
    # ansatz = qaoa_ansatz(cost_operator, reps=3, insert_barriers=True)
    # ansatz.draw("mpl")

    n = 4
    p = 0.5
    graph = nx.erdos_renyi_graph(n, p)
    # graph = nx.Graph()
    # graph.add_nodes_from([0, 1, 2, 3])
    # graph.add_edges_from([(0, 2), (1, 3)])
    # graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    problem_instance = {"graph": graph}
    plt.figure()
    nx.draw(graph, with_labels=True)
    plt.show()

    paulis = build_maxclique_mis_paulis(problem_instance)
    print(paulis)

    cost_hamiltonian = SparsePauliOp.from_sparse_list(paulis, graph.number_of_nodes())
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)

    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2, flatten=True)
    circuit.measure_all()
    circuit = pm.run(circuit)
    print(f"QAOAAnsatz Operations: {circuit.decompose("rzz").count_ops()}")
    print(f"QAOAAnsatz Depth: {circuit.decompose("rzz").depth()}")
    # circuit.decompose("rzz").draw("mpl", fold=50, scale=0.5)
    # plt.show()

    my_circuit_dfs = myQAOAAnsatz(cost_hamiltonian=cost_hamiltonian, reps=2, problem_instance=problem_instance, use_coloring=True, optimize_cnots=True, use_dfs=True)
    my_circuit_dfs.measure_all()
    my_circuit_dfs = pm.run(my_circuit_dfs)
    print(f"myQAOAAnsatzDFS Operations: {my_circuit_dfs.count_ops()}")
    print(f"myQAOAAnsatzDFS Depth: {my_circuit_dfs.depth()}")
    # my_circuit_dfs.draw("mpl", fold=50, scale=0.5)
    # plt.show()

    # my_circuit_no_optim = myQAOAAnsatz(cost_hamiltonian=cost_hamiltonian, reps=2, use_coloring=False, problem_instance=problem_instance, optimize_cnots=False)
    # my_circuit_no_optim.measure_all()
    # my_circuit_no_optim = pm.run(my_circuit_no_optim)
    # print(f"myQAOAAnsatz Operations (no optim): {my_circuit_no_optim.count_ops()}")
    # print(f"myQAOAAnsatz Depth (no optim): {my_circuit_no_optim.depth()}")
    # my_circuit_no_optim.draw("mpl", fold=50, scale=0.5)
    # plt.show()

    # my_circuit_coloring = myQAOAAnsatz(cost_hamiltonian=cost_hamiltonian, reps=2, use_coloring=True, problem_instance=problem_instance, optimize_cnots=False)
    # my_circuit_coloring.measure_all()
    # my_circuit_coloring = pm.run(my_circuit_coloring)
    # print(f"myQAOAAnsatz Operations (coloring): {my_circuit_coloring.count_ops()}")
    # print(f"myQAOAAnsatz Depth (coloring): {my_circuit_coloring.depth()}")
    # my_circuit_coloring.draw("mpl", fold=50, scale=0.5)
    # plt.show()

    # my_circuit_coloring_cnots = myQAOAAnsatz(cost_hamiltonian=cost_hamiltonian, reps=2, use_coloring=True, problem_instance=problem_instance, optimize_cnots=True)
    # my_circuit_coloring_cnots.measure_all()
    # my_circuit_coloring_cnots = pm.run(my_circuit_coloring_cnots)
    # print(f"myQAOAAnsatz Operations (coloring + cnots): {my_circuit_coloring_cnots.count_ops()}")
    # print(f"myQAOAAnsatz Depth (coloring + cnots): {my_circuit_coloring_cnots.depth()}")
    # my_circuit_coloring_cnots.draw("mpl", fold=50, scale=0.5)
    # plt.show()

    # qaoaansatz = quick_run_qaoa(
    #     problem_instance, build_maxclique_mis_paulis, iters=1, num_layers=2, num_shots=4096, ansatz_circuit=myQAOAAnsatz)
    ideal_sv = quick_run_qaoa(
        problem_instance, build_maxclique_mis_paulis, iters=1, use_noisy_optimizer=False, num_layers=1, num_shots=512, ansatz_circuit=lambda ch, reps: myQAOAAnsatz(
                cost_hamiltonian=ch, 
                reps=reps, 
                use_coloring=True, 
                problem_instance=problem_instance,
                optimize_cnots=True
            ),
            return_statevectors=True
        )
    noisy_sv = quick_run_qaoa(
        problem_instance, build_maxclique_mis_paulis, iters=1, use_noisy_optimizer=True, num_layers=1, num_shots=512, ansatz_circuit=lambda ch, reps: myQAOAAnsatz(
                cost_hamiltonian=ch, 
                reps=reps, 
                use_coloring=True, 
                problem_instance=problem_instance,
                optimize_cnots=True
            ),
            return_statevectors=True
        )
    print(ideal_sv)
    print(noisy_sv)
    # for res in qaoaansatz[0]:
    #     print(res.get_statevector())
    #     counts = res.data.meas.get_int_counts()
    #     plt.figure()
    #     plt.bar(*zip(*counts.items()))
    #     plt.show()
    #     print(counts)

    # print(test_graph_qaoa(
    #     problem_instance, 
    #     build_maxclique_mis_paulis, 
    #     validate_solutions=lambda solutions: validate_max_clique_solutions(problem_instance, solutions), 
    #     num_layers=2, 
    #     iters=1, 
    #     use_noisy_optimizer=False,
    #     # ansatz_circuit=lambda ch, reps: myQAOAAnsatz(
    #     #         cost_hamiltonian=ch, 
    #     #         reps=reps, 
    #     #         use_coloring=True, 
    #     #         problem_instance=problem_instance,
    #     #         optimize_cnots=True,
    #     #     )
    #     # )
    #     ansatz_circuit=QAOAAnsatz))
    
    # print(test_graph_qaoa(
    #     problem_instance, 
    #     build_maxclique_mis_paulis, 
    #     validate_solutions=lambda solutions: validate_max_clique_solutions(problem_instance, solutions), 
    #     num_layers=2, 
    #     iters=1, 
    #     use_noisy_optimizer=False,
    #     ansatz_circuit=lambda ch, reps: myQAOAAnsatz(
    #             cost_hamiltonian=ch, 
    #             reps=reps, 
    #             use_coloring=True, 
    #             problem_instance=problem_instance,
    #             optimize_cnots=True,
    #             use_dfs=True
    #         )
    #     ))
    #     # ansatz_circuit=QAOAAnsatz))
