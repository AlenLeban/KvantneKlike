from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister, generate_preset_pass_manager, transpile
from qiskit.circuit.library import grover_operator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import AerSimulator

def is_k_clique_oracle(graph, k):
    n = graph.number_of_nodes()
    q_verts = QuantumRegister(n, "qv")
    # q_edges = QuantumRegister(int(n*n - (1 + n)/2 * n), "qe")
    q_size = AncillaRegister(int(np.ceil(np.log2(n+1))), "a_size")
    q_edge_ancilla = AncillaRegister(int(n*n - (1 + n)/2 * n), "ae_anc")
    q_isclique = AncillaRegister(1, "a_isclique")
    q_isk = AncillaRegister(1, "a_isk")
    q_ans = AncillaRegister(1, "a_ans")
    q_mcx_ancilla = AncillaRegister(len(q_edge_ancilla), "a_mcx")
    qc = QuantumCircuit(q_verts, q_edge_ancilla, q_isclique, q_size, q_isk, q_ans, q_mcx_ancilla)
    pair_to_edge_offset = dict()
    for i in range(n):
        for j in range(n):
            if i>=j:
                continue
            pair_to_edge_offset[(i,j)] = len(pair_to_edge_offset)

    print(pair_to_edge_offset)

    # encoding graph edges
    # for edge in graph.edges():
    #     n1 = edge[0] if edge[0] < edge[1] else edge[1]
    #     n2 = edge[1] if n1 == edge[0] else edge[0]
    #     edge_qubit = q_edges[pair_to_edge_offset[(n1, n2)]]
    #     qc.x(edge_qubit)

    # checking connectivity between candidate vertices
    for n1 in graph.nodes():
        for n2 in graph.nodes():
            if n1 >= n2:
                continue
            has_edge = graph.has_edge(n1, n2)
            if has_edge:
                continue
            ancilla_qubit = q_edge_ancilla[pair_to_edge_offset[(n1, n2)]]
            qc.mcx([n1, n2], ancilla_qubit)

    # count vertices
    for v in range(n):
        for i in reversed(range(1, len(q_size))):
            qc.mcx([q_verts[v]] + list(q_size[:i]), q_size[i])
        qc.cx(q_verts[v], q_size[0])

    # check if vertex count is k
    qc.mcx(q_size, q_isk, ctrl_state=format(k, f'0{len(q_size)}b'))

    # check clique violations
    qc.x(q_edge_ancilla)
    qc.mcx(q_edge_ancilla, q_isclique, ancilla_qubits=q_mcx_ancilla, mode="v-chain")

    qc.x(q_ans)
    qc.h(q_ans)
    qc.mcx([q_isclique, q_isk], q_ans)
    qc.h(q_ans)
    qc.x(q_ans)

    # uncompute

    # qc.mcx(q_edge_ancilla, q_isclique)
    
    qc.x(q_edge_ancilla)
    qc.mcx(q_size, q_isk, ctrl_state=format(k, f'0{len(q_size)}b'))

    for v in reversed(range(n)):
        qc.cx(q_verts[v], q_size[0])
        for i in range(1, len(q_size)):
            qc.mcx([q_verts[v]] + list(q_size[:i]), q_size[i])

    for n1 in reversed(list(graph.nodes())):
        for n2 in reversed(list(graph.nodes())):
            if n1 >= n2:
                continue
            has_edge = graph.has_edge(n1, n2)
            if has_edge:
                continue
            ancilla_qubit = q_edge_ancilla[pair_to_edge_offset[(n1, n2)]]
            qc.mcx([n1, n2], ancilla_qubit)

    # for edge in graph.edges():
    #     n1 = edge[0] if edge[0] < edge[1] else edge[1]
    #     n2 = edge[1] if n1 == edge[0] else edge[0]
    #     edge_qubit = q_edges[pair_to_edge_offset[(n1, n2)]]
    #     qc.x(edge_qubit)

    return qc

if __name__ == "__main__":
    test_graph = nx.Graph()
    test_graph.add_nodes_from([0, 1, 2])
    test_graph.add_edges_from([(0, 1), (0, 2), (1, 2)])

    circuit = is_k_clique_oracle(test_graph, 3)

    grover_circuit_iter = grover_operator(circuit, reflection_qubits=list(range(4)))
    # grover_circuit_iter.draw("mpl")

    iterations = 2
    grover_circuit = QuantumCircuit(grover_circuit_iter.num_qubits, test_graph.number_of_nodes())
    grover_circuit.h(range(test_graph.number_of_nodes()))
    grover_circuit.compose(grover_circuit_iter.power(iterations), inplace=True)
    grover_circuit.measure(range(test_graph.number_of_nodes()), range(test_graph.number_of_nodes()))
    # grover_circuit.draw("mpl")

    grover_circuit_t = transpile(grover_circuit, basis_gates=["rz", "sx", "cx"])
    print(f"Transpiled depth: {grover_circuit_t.depth()}")
    # circuit.draw("mpl", scale=1, fold=50, interactive=True)
    # plt.show()
    # print(grover_circuit.num_qubits)

    grover_use_simulator = True
    if not grover_use_simulator:
        service = QiskitRuntimeService()

        backend = service.least_busy(
            operational=True, simulator=False, min_num_qubits=127
        )
    else:
        backend = FakeBrisbane()
        # backend = AerSimulator()
    print(backend)

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = 1024
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    candidate_circuit = pm.run(grover_circuit)

    print("Original qubits:", grover_circuit.num_qubits)
    print("Transpiled qubits:", candidate_circuit.num_qubits)
    print("Backend qubits:", backend.num_qubits)

    print("depth:", candidate_circuit.depth())
    print("ops:", candidate_circuit.count_ops())
    print("qubits:", candidate_circuit.num_qubits)



    pub = (candidate_circuit, )
    job = sampler.run([pub])
    print(job.result()[0].data.c)
    counts_int = job.result()[0].data.c.get_int_counts()
    counts_bin = job.result()[0].data.c.get_counts()
    shots = sum(counts_int.values())
    final_distribution_int = {key: val / shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
    print(final_distribution_int)
    print(counts_int)
    print(counts_bin)
    plt.rcParams.update({"font.size": 10})
    final_bits = final_distribution_bin
    values = np.abs(list(final_bits.values()))
    top_4_values = sorted(values, reverse=True)[:4]
    positions = []
    for value in top_4_values:
        positions.append(np.where(values == value)[0])
    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks(rotation=90)
    plt.title("Result Distribution")
    plt.xlabel("Bitstrings (reversed)")
    plt.ylabel("Probability")
    ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")
    for p in positions:
        ax.get_children()[int(p[0])].set_color("tab:purple")
    plt.show()

# candidate_circuit.draw("mpl")