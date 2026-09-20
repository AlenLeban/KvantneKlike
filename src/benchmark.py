from bayes_opt import BayesianOptimization
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

from annealing import qa_k_clique_bqm, qa_max_clique_bqm, test_problem_sizes_qa
from qaoa import build_kclique_paulis_mis, build_maxclique_mis_paulis, test_problem_sizes_qaoa
from helper_functions import helper_validate_max_clique_solutions
from utils import er_max_clique_size, generate_k_clique_instance, generate_random_graph_instance, validate_k_clique_solutions, validate_max_clique_solutions
from bayes_opt import acquisition

def benchmark_problem_sizes(problem_sizes, output_filename, problem, validate_solutions, instance_generator, num_graphs=3, iters_per_graph=5, method="QAOA", append_graphs=False, append_problem_sizes=False,
                            use_noise=False, intermediate_results_filename=None, max_workers=10, return_eigenenergies=False):
    out_dict = dict()
    if append_graphs or append_problem_sizes:
        try:
            with open("benchmarkResults/" + output_filename, "r") as f:
                out_dict = json.load(f)
        except:
            print("Could not open file to append graphs / problem sizes. Setting to false")
            append_graphs=False
            append_problem_sizes=False
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
                                                max_workers=max_workers,
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
                                max_workers=max_workers,
                                use_noise=use_noise,
                                return_eigenenergies=return_eigenenergies
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

def run_qa(schedule_params, sizes, generate_instance, instance_count, problem, validate_solutions, max_workers=12, return_eigenenergies=False):


    # results = test_problem_sizes_qa(
    #     # [{"n": 9, "p": 0.4, "k": None, "schedule": "lin_interp", "schedule_params": list(params.values())}],
    #     # [{"n": 9, "p": 0.4, "k": None, "schedule": "lin_interp", "schedule_params": offsets}],
    #     sizes=[{"n": 8, "p": 0.4, "k": er_max_clique_size(8, 0.4)-1, "schedule": "cubic_interp", "schedule_params": schedule_params, "steps":30}],
    #     generate_instance=generate_k_clique_instance,
    #     instance_count=100,
    #     problem=qa_k_clique_bqm,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     iters=1,
    #     max_workers=12,
    #     return_eigenenergies=return_eigenenergies
    # )

    results = test_problem_sizes_qa(
        # [{"n": 9, "p": 0.4, "k": None, "schedule": "lin_interp", "schedule_params": list(params.values())}],
        # [{"n": 9, "p": 0.4, "k": None, "schedule": "lin_interp", "schedule_params": offsets}],
        sizes=[{**s, "schedule_params": schedule_params} for s in sizes],
        generate_instance=generate_instance,
        instance_count=instance_count,
        problem=problem,
        validate_solutions=validate_solutions,
        iters=1,
        max_workers=max_workers,
        return_eigenenergies=return_eigenenergies
    )

    # foreach problem size, foreach instance [success prob and energies]
    # probs = [p["success_probability"] for p in results]
    total_prob = 0
    probs = []
    for r in results[0]:
        if return_eigenenergies:
            probs.append(r[0]["success_probability"])
        else:
            probs.append(r["success_probability"])
    if return_eigenenergies:
        energies_over_time = [p[1] for p in results[0]]
    else:
        energies_over_time = None
    return probs, energies_over_time




def qa_bo_optimize_schedule(optimizer_state_filename, sizes, generate_instance, instance_count, problem, validate_solutions, max_workers=12, return_eigenenergies=False, n_params=6, n_iter=50, init_points=10, probe_points=[], **kwargs):

    def bo_run_qa(**params):

        # constraint that each next value must be larger
        if sum(list(params.values())) > 1:
            return 0

        offsets = [0]
        for v in list(params.values()):
            current_sum = offsets[-1] + v
            if current_sum > 1 or current_sum < 0:
                return 0
            offsets.append(offsets[-1] + v)
        offsets = offsets[1:]

        probs, _ = run_qa(offsets, sizes, generate_instance, instance_count, problem, validate_solutions, max_workers, return_eigenenergies)
        avg_prob = sum(probs) / len(probs)
        return avg_prob

    acquisition_function = acquisition.ExpectedImprovement(xi=kwargs.get("alpha", 0.02))
    pbounds = {f"p{i}": (0, 0.6) for i in range(n_params)}
    print(pbounds)
    optimizer = BayesianOptimization(
        f=bo_run_qa,
        pbounds=pbounds,
        verbose=2,
        allow_duplicate_points=True,
        acquisition_function=acquisition_function
    )
    try:
        optimizer.load_state(optimizer_state_filename)
    except:
        print(f"Could not load initial state of optimizer from {optimizer_state_filename}. Starting fresh.")

    for point in probe_points:
        optimizer.probe(point)

    optimizer.maximize(n_iter=n_iter, init_points=init_points)
    optimizer.save_state(optimizer_state_filename)
    print(optimizer.max)

    # optimize_schedule_power(optimizer.max["params"])


    # experiments


    # filename = "qa_gap_results_kclique_boquadratic.json"
    # probs, energies = run_qa(np.cumsum(list(optimizer.max["params"].values())))


    # filename = "qa_gap_results_kclique_linear.json"
    # probs, energies = run_qa(np.linspace(0, 1, n_params+2)[1:-1])

    # optimizer.load_state("benchmarkResults/schedule_optimization/optimizer_state_maxclique_linear_n8_p4_6param.json")
    # filename = "qa_gap_results_maxclique_lininterp.json"
    # probs, energies = run_qa(np.cumsum(list(optimizer.max["params"].values())))

    # optimizer.load_state("benchmarkResults/schedule_optimization/optimizer_state_maxclique_cubic_n8_p4_6param.json")
    # filename = "qa_gap_results_maxclique_boquadratic.json"
    # probs, energies = run_qa(np.cumsum(list(optimizer.max["params"].values())))

    # filename = "qa_gap_results_maxclique_linear.json"
    # probs, energies = run_qa(np.linspace(0, 1, n_params+2)[1:-1])

    # filename = "qa_gap_results_maxclique_boquadratic.json"
    # probs, energies = run_qa(np.cumsum(list(optimizer.max["params"].values())))

    # append_graphs = True
    # avg_probs = np.mean(probs)
    # energies_np = np.array(energies)
    # out_data = []
    # for i in range(len(energies)):
    #     ground_state_degeneracy = sum([e - energies[i][-1][0] < 1e-03 for e in energies[i][-1]])
    #     # print(ground_state_degeneracy)
    #     if ground_state_degeneracy == len(energies_np[i,0]):
    #         continue
    #     # print(energies[i][-1])
    #     # print(energies_np[i,:,ground_state_degeneracy] - energies_np[i,:,0])
    #     min_gap = np.min(energies_np[i,:,ground_state_degeneracy] - energies_np[i,:,0])
    #     # print(min_gap)
    #     out_data.append({
    #         "min_gap": min_gap,
    #         "prob": probs[i]
    #     })
    # # print(energies)


    # out_dict = None
    # if append_graphs:
    #     try:
    #         with open(f"benchmarkResults/min_gap/{filename}", "r") as f:
    #             out_dict = json.load(f)
    #     except:
    #         print(f"Unable to open benchmarkResults/min_gap/f{filename} to append graphs")

    # with open(f"benchmarkResults/min_gap/{filename}", "w") as f:
    #     if out_dict != None:
    #         out_data.extend(out_dict["samples"])
    #     json.dump({
    #         "samples": out_data
    #     }, f)

    # nrows = int(np.ceil(np.sqrt(len(energies))))
    # ncols = int(np.ceil(len(energies)/nrows))
    # print(nrows, ncols)
    # fig, ax = plt.subplots(nrows, ncols, squeeze=False)
    # for i,e in enumerate(energies):
    #     instance_energies = np.array(e)
    #     instance_energies = instance_energies - np.reshape(instance_energies[:,0], (len(instance_energies), 1))
    #     for j in range(len(instance_energies[0])):
    #         ax[i//ncols, i%ncols].plot(np.linspace(0, 1, len(instance_energies)), instance_energies[:,j])
    # plt.plot()
    # plt.show()

    # plt.figure()
    # cumsums = np.cumsum(list(optimizer.max["params"].values()))
    # cs = CubicSpline(np.linspace(0, 1, len(cumsums)+2), [0, *cumsums, 1])
    # plt.plot(np.linspace(0, 1, 30), cs(np.linspace(0, 1, 30)))

    # lininterp

    # cumsums = np.cumsum(list(optimizer.max["params"].values()))
    # plt.plot(np.linspace(0, 1, 8), [0, *cumsums, 1])
    # plt.show()


if __name__ == "__main__":    
    pass