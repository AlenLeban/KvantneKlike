from annealing import qa_k_clique_bqm, qa_max_clique_bqm
from benchmark import benchmark_problem_sizes, helper_validate_k_clique_solutions, helper_validate_max_clique_solutions
from qaoa import build_maxclique_mis_paulis
from utils import er_max_clique_size, generate_k_clique_instance, generate_random_graph_instance


if __name__ == "__main__":
    # QAOA

    # K-CLIQUE

    # test qaoa on findinig k-clique, different n
    # benchmark_problem_sizes(
    #     [ {"n": n, "p": 0.4, "k": int(er_max_clique_size(n, 0.4)-2), "iters_per_graph" : 40} for n in range(6, 20, 2)],
    #     "qa_results_kclique_n_noisy.json",
    #     problem=qa_k_clique_bqm,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     instance_generator=generate_k_clique_instance,
    #     num_graphs=50,
    #     method="QA",
    #     use_noise=True
    # )

    # test qaoa on findinig k-clique, different iters per graph
    # benchmark_problem_sizes(
    #     [ {"n": n, "p": 0.4, "k": int(er_max_clique_size(n, 0.4)-1), "iters_per_graph" : iters} for iters in range(1, 22, 4) for n in range(6, 200, 50)],
    #     "qa_results_kclique_iters_n.json",
    #     problem=qa_k_clique_bqm,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     instance_generator=generate_k_clique_instance,
    #     num_graphs=300,
    #     method="QA",
    #     use_noise=False
    # )

    # test qaoa on findinig k-clique, different iters per graph NOISY
    # benchmark_problem_sizes(
    #     [ {"n": n, "p": 0.4, "k": int(er_max_clique_size(n, 0.4)-1), "iters_per_graph" : iters} for iters in range(1, 22, 4) for n in range(6, 200, 50)],
    #     "qa_results_kclique_iters_n_noisy.json",
    #     problem=qa_k_clique_bqm,
    #     validate_solutions=helper_validate_k_clique_solutions,
    #     instance_generator=generate_k_clique_instance,
    #     num_graphs=300,
    #     method="QA",
    #     use_noise=True
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

    # test qa on finding max-clique, different iters per graph
    benchmark_problem_sizes(
        [ {"n": n, "p": 0.4, "k": None, "iters_per_graph" : iters} for iters in range(1, 22, 4) for n in range(6, 200, 50)],
        "qa_results_maxclique_iters_n.json",
        problem=qa_max_clique_bqm,
        validate_solutions=helper_validate_max_clique_solutions,
        instance_generator=generate_random_graph_instance,
        num_graphs=150,
        method="QA",
        append_graphs=True,
        use_noise=False
    )

    # test qa on finding max-clique, different iters per graph (NOISY)
    benchmark_problem_sizes(
        [ {"n": n, "p": 0.4, "k": None, "iters_per_graph" : iters} for iters in range(1, 22, 4) for n in range(6, 200, 50)],
        "qa_results_maxclique_iters_n_noisy.json",
        problem=qa_max_clique_bqm,
        validate_solutions=helper_validate_max_clique_solutions,
        instance_generator=generate_random_graph_instance,
        num_graphs=200,
        method="QA",
        use_noise=True
    )

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
    # benchmark_problem_sizes(
    #     [ {"n": 10, "p": float(round(p*1000)/1000), "k": None, "iters_per_graph" : 1, "layers": 2} for p in np.arange(0.1, 0.8, 0.3)],
    #     "qaoa_results_maxclique_p_oneiter_noisy.json",
    #     problem=build_maxclique_mis_paulis,
    #     validate_solutions=helper_validate_max_clique_solutions,
    #     instance_generator=generate_random_graph_instance,
    #     num_graphs=10,
    #     method="QAOA",
    #     append_graphs=False,
    #     use_noisy_optimizer=True
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