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
from qaoa import test_problem_sizes_qaoa
from utils import er_max_clique_size, generate_k_clique_instance, validate_k_clique_solutions


def benchmark_problem_sizes(problem_sizes, output_filename, problem, validate_solutions, num_graphs=3, iters_per_graph=5, method="QAOA"):
    out_dict = dict()
    out_dict["problem_sizes"] = problem_sizes
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
            
        validation_results, depths_per_size, evaluations_per_size = test_problem_sizes_qaoa(problem_sizes, 
                                                generate_k_clique_instance,
                                                num_graphs, 
                                                problem=problem, 
                                                validate_solutions=validate_solutions,
                                                iters=iters
                                                )
        print(validation_results)
        out_dict["results_per_size"] = validation_results
        out_dict["depths_per_size"] = depths_per_size
        out_dict["evaluations_per_size"] = evaluations_per_size
    
    elif method == "QA":
        validation_results = test_problem_sizes_qa(problem_sizes, 
                                generate_k_clique_instance,
                                num_graphs, 
                                problem=problem, 
                                validate_solutions=validate_solutions,
                                iters=iters
                                )
        out_dict["results_per_size"] = validation_results

    with open("benchmarkResults/" + output_filename, "w") as f:
        json.dump(out_dict, f)
    return out_dict

if __name__ == "__main__":

    # test qa on findinig k-clique
    # benchmark_graph_size_results = benchmark_problem_sizes(
    #     [ {"n": n, "p": 0.4, "k": int(er_max_clique_size(n, 0.4))-1, "iters_per_graph" : 50} for n in range(6, 25, 2)],
    #     "qa_results_kclique_n.json",
    #     problem=qa_k_clique_bqm,
    #     validate_solutions=lambda g,x,p: validate_k_clique_solutions(g, x, p["k"]),
    #     num_graphs=30,
    #     method="QA"
    # )

    pass
