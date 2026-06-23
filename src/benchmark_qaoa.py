import numpy as np
from benchmark import benchmark_problem_sizes, helper_validate_k_clique_solutions, helper_validate_max_clique_solutions
from qaoa import build_kclique_paulis_mis, build_maxclique_mis_paulis, test_problem_sizes_qaoa
from utils import er_max_clique_size, generate_k_clique_instance, generate_random_graph_instance, validate_k_clique_solutions, validate_max_clique_solutions

if __name__ == "__main__":    


    # QAOA

    # K-CLIQUE

    # test qaoa on findinig k-clique, different n
    # benchmark_problem_sizes(
    #     [ {"n": n, "p": 0.4, "k": er_max_clique_size(n, 0.4)-1, "iters_per_graph" : 4, "layers": 2} for n in range(6, 14, 1)],
    #     "qaoa_results_kclique_n.json",
    #     problem=build_kclique_paulis_mis,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     instance_generator=generate_k_clique_instance,
    #     # intermediate_results_filename="qaoa_results_kclique_n_intermediate.json",
    #     num_graphs=200,
    #     method="QAOA",
    #     append_graphs=False
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
    #     [ {"n": n, "p": 0.4, "k": int(er_max_clique_size(n, 0.4))-1, "iters_per_graph" : 2, "layers": layers} for layers in range(1, 6) for n in range(6, 13, 2)],
    #     "qaoa_results_kclique_layers_n_temp.json",
    #     problem=build_kclique_paulis_mis,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     instance_generator=generate_k_clique_instance,
    #     # intermediate_results_filename="qaoa_results_kclique_layers_n_temp_intermediate",
    #     num_graphs=100,
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
    #     [ {"n": 7, "p": float(round(p*1000)/1000), "k": 3, "iters_per_graph" : 1, "layers": 1} for p in np.arange(0.1, 0.8, 0.1)],
    #     "qaoa_results_kclique_p_noisy.json",
    #     problem=build_kclique_paulis_mis,
    #     instance_generator=generate_k_clique_instance,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     num_graphs=50,
    #     method="QAOA",
    #     append_graphs=True,
    #     use_noise=True,
    #     # intermediate_results_filename="qaoa_results_kclique_p_noisy_intermediate.json"
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
    #     [ {"n": n, "p": 0.4, "k": None, "iters_per_graph" : 4, "layers": 2} for n in range(6, 14, 1)],
    #     "qaoa_results_maxclique_n.json",
    #     problem=build_maxclique_mis_paulis,
    #     validate_solutions=helper_validate_max_clique_solutions,
    #     instance_generator=generate_random_graph_instance,
    #     # intermediate_results_filename="qaoa_results_kclique_n_intermediate.json",
    #     num_graphs=100,
    #     method="QAOA",
    #     append_graphs=True
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

    # test qaoa on finding max-clique, different p and n
    benchmark_problem_sizes(
        [ {"n": int(n), "p": float(round(p*1000)/1000), "k": None, "iters_per_graph" : 1, "layers": 1} for p in np.arange(0.1, 0.8, 0.1) for n in np.arange(6, 13, 2)],
        "qaoa_results_maxclique_p_n.json",
        problem=build_maxclique_mis_paulis,
        validate_solutions=helper_validate_max_clique_solutions,
        instance_generator=generate_random_graph_instance,
        num_graphs=200,
        method="QAOA",
        append_graphs=True,
        append_problem_sizes=False,
        use_noise=False
    )

    # test qaoa on finding max-clique, different p NOISY
    # benchmark_problem_sizes(
    #     [ {"n": 8, "p": float(round(p*1000)/1000), "k": None, "iters_per_graph" : 1, "layers": 1} for p in np.arange(0.1, 0.9, 0.2)],
    #     "qaoa_results_maxclique_p_oneiter_noisy.json",
    #     problem=build_maxclique_mis_paulis,
    #     validate_solutions=helper_validate_max_clique_solutions,
    #     instance_generator=generate_random_graph_instance,
    #     num_graphs=50,
    #     method="QAOA",
    #     append_graphs=False,
    #     use_noise=True
    # )

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
