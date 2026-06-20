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

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from neal import SimulatedAnnealingSampler
from dimod import Binary, ExactSolver
from dwave.samplers import PathIntegralAnnealingSampler
from tqdm import tqdm

from annealing import qa_k_clique_bqm, test_problem_sizes_qa
from qaoa import build_kclique_paulis_mis, build_maxclique_mis_paulis, test_problem_sizes_qaoa
from utils import er_max_clique_size, generate_k_clique_instance, generate_random_graph_instance, validate_k_clique_solutions, validate_max_clique_solutions


def benchmark_problem_sizes(problem_sizes, output_filename, problem, validate_solutions, instance_generator, num_graphs=3, iters_per_graph=5, method="QAOA", append_graphs=False, append_problem_sizes=False,
                            use_noise=False):
    out_dict = dict()
    if append_graphs or append_problem_sizes:
        with open("benchmarkResults/" + output_filename, "r") as f:
            out_dict = json.load(f)
    if append_problem_sizes:
        out_dict["problem_sizes"].extend(problem_sizes)
    elif not append_graphs:
        out_dict["problem_sizes"] = problem_sizes
    if append_graphs:
        out_dict["num_graphs"] += num_graphs
    elif not append_problem_sizes:
        out_dict["num_graphs"] = num_graphs
    iters = None

    if not "iters_per_graph" in problem_sizes[0]:
        for i in range(len(problem_sizes)):
            problem_sizes[i]["iters_per_graph"] = iters_per_graph
        out_dict["iters_per_graph"] = iters_per_graph
        iters = iters_per_graph

    if method == "QAOA":


        layers_default = 4
        layers = None
        if not "layers" in problem_sizes[0]:
            for i in range(len(problem_sizes)):
                problem_sizes[i]["layers"] = layers_default
            out_dict["layers"] = layers_default
            layers = layers_default
            
        validation_results, depths_per_size, evaluations_per_size = test_problem_sizes_qaoa(problem_sizes if not append_graphs else out_dict["problem_sizes"], 
                                                instance_generator,
                                                out_dict["num_graphs"] if not append_graphs else num_graphs, 
                                                problem=problem, 
                                                validate_solutions=validate_solutions,
                                                iters=iters,
                                                max_workers=10,
                                                layers=layers,
                                                use_noisy_optimizer=use_noise
                                                )
        print(validation_results)
        if append_graphs:
            for i in range(len(problem_sizes if not append_graphs else out_dict["problem_sizes"])):
                out_dict["results_per_size"][i].extend(validation_results[i])
                out_dict["depths_per_size"][i].extend(depths_per_size[i])
                out_dict["evaluations_per_size"][i].extend(evaluations_per_size[i])
        elif append_problem_sizes:
            out_dict["results_per_size"].extend(validation_results)
            out_dict["depths_per_size"].extend(depths_per_size)
            out_dict["evaluations_per_size"].extend(evaluations_per_size)
        else:
            out_dict["results_per_size"] = validation_results
            out_dict["depths_per_size"] = depths_per_size
            out_dict["evaluations_per_size"] = evaluations_per_size
    
    elif method == "QA":
        validation_results = test_problem_sizes_qa(problem_sizes if not append_graphs else out_dict["problem_sizes"], 
                                instance_generator,
                                out_dict["num_graphs"] if not append_graphs else num_graphs, 
                                problem=problem, 
                                validate_solutions=validate_solutions,
                                iters=iters,
                                max_workers=10,
                                use_noise=use_noise
                                )
        if append_graphs:
            for i in range(len(problem_sizes if not append_graphs else out_dict["problem_sizes"])):
                out_dict["results_per_size"][i].extend(validation_results[i])
        elif append_problem_sizes:
            out_dict["results_per_size"].extend(validation_results)
        else:
            out_dict["results_per_size"] = validation_results

    with open("benchmarkResults/" + output_filename, "w") as f:
        json.dump(out_dict, f)
    return out_dict

def helper_validate_k_clique_solutions(g, x, p):
    return validate_k_clique_solutions(g, x, p["k"])

def helper_validate_max_clique_solutions(g, x, p):
    return validate_max_clique_solutions(g, x)

if __name__ == "__main__":    



    # QAOA

    # K-CLIQUE

    # test qaoa on findinig k-clique, different n
    # benchmark_problem_sizes(
    #     [ {"n": n, "p": 0.4, "k": int(er_max_clique_size(n, 0.4))-1, "iters_per_graph" : 10} for n in range(6, 15, 2)],
    #     "qaoa_results_kclique_n.json",
    #     problem=build_kclique_paulis_mis,
    #     validate_solutions=lambda g,x,p: validate_k_clique_solutions(g, x, p["k"]),
    #     num_graphs=5,
    #     method="QAOA"
    # )

    # test qaoa on finding k-clique, different iters per graph
    # benchmark_problem_sizes(
    #     [ {"n": 12, "p": 0.4, "k": int(er_max_clique_size(12, 0.4))-1, "iters_per_graph" : iters} for iters in range(10, 50, 8)],
    #     "qaoa_results_kclique_iters.json",
    #     problem=build_kclique_paulis_mis,
    #     validate_solutions=lambda g,x,p: validate_k_clique_solutions(g, x, p["k"]),
    #     num_graphs=40,
    #     method="QAOA"
    # )

    # test qaoa on finding k-clique, different num layers per graph size
    # benchmark_problem_sizes(
    #     [ {"n": n, "p": 0.4, "k": int(er_max_clique_size(n, 0.4))-1, "iters_per_graph" : int((n**1.7)/2), "layers": layers} for layers in range(1, 5) for n in range(6, 11, 4)],
    #     "qaoa_results_kclique_layers_n.json",
    #     problem=build_kclique_paulis_mis,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     num_graphs=20,
    #     method="QAOA",
    #     append_graphs=True
    # )

    # test qaoa on finding k-clique, different p
    # benchmark_problem_sizes(
    #     [ {"n": 10, "p": p, "k": 5, "iters_per_graph" : 20, "layers": 2} for p in np.arange(0.1, 0.8, 0.1)],
    #     "qaoa_results_kclique_p.json",
    #     problem=build_kclique_paulis_mis,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     num_graphs=40,
    #     method="QAOA",
    #     append_graphs=True
    # )

    # test qaoa on finding k-clique, different p (NOISY)
    # benchmark_problem_sizes(
    #     [ {"n": 7, "p": float(round(p*1000)/1000), "k": 3, "iters_per_graph" : 1, "layers": 1} for p in np.arange(0.1, 0.8, 0.2)],
    #     "qaoa_results_kclique_p_noisy.json",
    #     problem=build_kclique_paulis_mis,
    #     instance_generator=generate_k_clique_instance,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     num_graphs=20,
    #     method="QAOA",
    #     # append_graphs=True,
    #     use_noise=True
    # )

    # test qaoa on finding k-clique, different k targets
    # benchmark_problem_sizes(
    #     [ {"n": 10, "p": 0.4, "k": k, "iters_per_graph" : 20, "layers": 2} for k in range(2, 9)],
    #     "qaoa_results_kclique_k.json",
    #     problem=build_kclique_paulis_mis,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     num_graphs=50,
    #     method="QAOA",
    #     append_graphs=True
    # )


    # MAX-CLIQUE

    # test qaoa on findinig max-clique, different n
    # benchmark_problem_sizes(
    #     [ {"n": n, "p": 0.4, "k": None, "iters_per_graph" : 1, "layers": 2} for n in range(11, 14, 2)],
    #     # [ {"n": 16, "p": 0.4, "k": None, "iters_per_graph" : 1, "layers": 2} ],
    #     "qaoa_results_maxclique_n_noisy.json",
    #     problem=build_maxclique_mis_paulis,
    #     validate_solutions=helper_validate_max_clique_solutions,
    #     instance_generator=generate_random_graph_instance,
    #     num_graphs=10,
    #     method="QAOA",
    #     # append_graphs=True,
    #     append_problem_sizes=True,
    #     use_noisy_optimizer=True
    # )

    # test qaoa on finding max-clique, different iters per graph
    # benchmark_problem_sizes(
    #     [ {"n": 10, "p": 0.4, "k": None, "iters_per_graph" : iters, "layers": 2} for iters in range(1, 20, 3)],
    #     # [ { "n": 12, "p": 0.4, "k": 5, "iters_per_graph": 27, "layers": 2 }, { "n": 12, "p": 0.4, "k": 5, "iters_per_graph": 30, "layers": 2 }, { "n": 12, "p": 0.4, "k": 5, "iters_per_graph": 33, "layers": 2 } ],
    #     "qaoa_results_maxclique_iters.json",
    #     problem=build_maxclique_mis_paulis,
    #     validate_solutions=helper_validate_max_clique_solutions,
    #     instance_generator=generate_random_graph_instance,
    #     num_graphs=20,
    #     method="QAOA",
    #     append_graphs=True
    # )

    # test qaoa on finding max-clique, different iters per graph NOISY
    # benchmark_problem_sizes(
    #     [ {"n": 7, "p": 0.4, "k": None, "iters_per_graph" : iters, "layers": 2} for iters in range(1, 7, 3)],
    #     # [ { "n": 12, "p": 0.4, "k": 5, "iters_per_graph": 27, "layers": 2 }, { "n": 12, "p": 0.4, "k": 5, "iters_per_graph": 30, "layers": 2 }, { "n": 12, "p": 0.4, "k": 5, "iters_per_graph": 33, "layers": 2 } ],
    #     "qaoa_results_maxclique_iters_noisy.json",
    #     problem=build_maxclique_mis_paulis,
    #     validate_solutions=helper_validate_max_clique_solutions,
    #     instance_generator=generate_random_graph_instance,
    #     num_graphs=10,
    #     method="QAOA",
    #     append_graphs=False,
    #     use_noisy_optimizer=True
    # )

    # test qaoa on finding max-clique, different num layers per graph size
    # benchmark_problem_sizes(
    #     [ {"n": n, "p": 0.4, "k": int(er_max_clique_size(n, 0.4)), "iters_per_graph" : 4, "layers": layers} for layers in range(1, 5) for n in range(6, 11, 4)],
    #     "qaoa_results_maxclique_layers_n.json",
    #     problem=build_maxclique_mis_paulis,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     num_graphs=160,
    #     method="QAOA",
    #     append_graphs=True
    # )

    # test qaoa on finding max-clique, different graph densities
    benchmark_problem_sizes(
        [ {"n": 8, "p": float(round(p*1000)/1000), "k": None, "iters_per_graph" : 1, "layers": 1} for p in np.arange(0.1, 0.9, 0.2)],
        "qaoa_results_maxclique_p_oneiter.json",
        problem=build_maxclique_mis_paulis,
        validate_solutions=helper_validate_max_clique_solutions,
        instance_generator=generate_random_graph_instance,
        num_graphs=50,
        method="QAOA",
        append_graphs=False,
        use_noise=False
    )

    # test qaoa on finding max-clique, different graph densities NOISY
    benchmark_problem_sizes(
        [ {"n": 8, "p": float(round(p*1000)/1000), "k": None, "iters_per_graph" : 1, "layers": 1} for p in np.arange(0.1, 0.9, 0.2)],
        "qaoa_results_maxclique_p_oneiter_noisy.json",
        problem=build_maxclique_mis_paulis,
        validate_solutions=helper_validate_max_clique_solutions,
        instance_generator=generate_random_graph_instance,
        num_graphs=50,
        method="QAOA",
        append_graphs=False,
        use_noise=True
    )

    # test qaoa on finding k-clique, different k targets
    # benchmark_problem_sizes(
    #     [ {"n": 10, "p": 0.4, "k": max(int(er_max_clique_size(10, 0.4))-i, 1), "iters_per_graph" : 20, "layers": 3} for i in range(1, 4)],
    #     "qaoa_results_kclique_k.json",
    #     problem=build_kclique_paulis_mis,
    #     validate_solutions=lambda g,x,p: validate_k_clique_solutions(g, x, p["k"]),
    #     num_graphs=10,
    #     method="QAOA",
    #     append_graphs=True
    # )

    pass
