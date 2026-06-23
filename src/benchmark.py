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
                            use_noise=False, intermediate_results_filename=None):
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

        def intermediate_results_callback(validation_results, depths_per_size, evaluations_per_size):
            out_dict_intermediate = out_dict.copy()
            if append_graphs:
                for i in range(len(validation_results if not append_graphs else out_dict_intermediate["problem_sizes"])):
                    out_dict_intermediate["results_per_size"][i].extend(validation_results[i])
                    out_dict_intermediate["depths_per_size"][i].extend(depths_per_size[i])
                    out_dict_intermediate["evaluations_per_size"][i].extend(evaluations_per_size[i])
            elif append_problem_sizes:
                out_dict_intermediate["results_per_size"].extend(validation_results)
                out_dict_intermediate["depths_per_size"].extend(depths_per_size)
                out_dict_intermediate["evaluations_per_size"].extend(evaluations_per_size)
            else:
                out_dict_intermediate["results_per_size"] = validation_results
                out_dict_intermediate["depths_per_size"] = depths_per_size
                out_dict_intermediate["evaluations_per_size"] = evaluations_per_size
            with open("benchmarkResults/" + intermediate_results_filename, "w") as f:
                json.dump(out_dict_intermediate, f)

            
        validation_results, depths_per_size, evaluations_per_size = test_problem_sizes_qaoa(problem_sizes if not append_graphs else out_dict["problem_sizes"], 
                                                instance_generator,
                                                out_dict["num_graphs"] if not append_graphs else num_graphs, 
                                                problem=problem, 
                                                validate_solutions=validate_solutions,
                                                iters=iters,
                                                max_workers=10,
                                                layers=layers,
                                                use_noisy_optimizer=use_noise,
                                                intermediate_results_callback=intermediate_results_callback if intermediate_results_filename else None
                                                )
        
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
    return validate_k_clique_solutions(g, x)

def helper_validate_max_clique_solutions(g, x, p):
    return validate_max_clique_solutions(g, x)

if __name__ == "__main__":    
    pass