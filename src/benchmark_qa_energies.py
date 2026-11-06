from benchmark import qa_bo_optimize_schedule, qa_test_energies
from annealing import qa_k_clique_bqm, qa_max_clique_bqm
from helper_functions import helper_validate_k_clique_solutions, helper_validate_max_clique_solutions
import numpy as np
from utils import generate_k_clique_instance, generate_kcore_graph_instance, generate_random_graph_instance, generate_random_instance_from_file, generate_specific_instance_from_file

if __name__ == "__main__":

    # test qa maxclique, different B
    # for b in [1, 2, 4, 8, 16]:
    #     qa_test_energies(f"benchmarkResults/energies/qa_energies_results_maxclique_B{b}.json", 
    #                             [{"n": 8, "p": 0.4, "k": None, "schedule": "linear_power", "B": b, "T":20, "steps":60}],
    #                             generate_instance=generate_random_graph_instance,
    #                             instance_count=50,
    #                             problem=qa_max_clique_bqm,
    #                             validate_solutions=helper_validate_max_clique_solutions,
    #                             schedule_params=[1.2],
    #                             iters=1,
    #                             max_workers=10,
    #                             append_graphs=True
    #                         )

    # test qa kclique, different B
    # for b in [8, 16]:
    #     qa_test_energies(f"benchmarkResults/energies/qa_energies_results_kclique_B{b}.json", 
    #                             [{"n": 8, "p": 0.4, "k": 3, "schedule": "linear_power", "B": b, "T":20, "steps":60}],
    #                             generate_instance=generate_k_clique_instance,
    #                             instance_count=50,
    #                             problem=qa_k_clique_bqm,
    #                             validate_solutions=helper_validate_k_clique_solutions,
    #                             schedule_params=[1.2],
    #                             iters=1,
    #                             max_workers=10,
    #                             append_graphs=True
    #                         )

    # test qa maxclique, sifferent kcore size
    # for kcore in [1, 2, 3, 4]:
    #     qa_test_energies(f"benchmarkResults/energies/qa_energies_results_maxclique_kcore{kcore}.json", 
    #                                 [{"n": 8, "p": 0.5, "k": None, "schedule": "linear_power", "B": 2, "T":20, "steps":30, "k_score_size": kcore}],
    #                                 generate_instance=generate_kcore_graph_instance,
    #                                 instance_count=150,
    #                                 problem=qa_max_clique_bqm,
    #                                 validate_solutions=helper_validate_max_clique_solutions,
    #                                 schedule_params=[1.2],
    #                                 iters=1,
    #                                 max_workers=10,
    #                                 append_graphs=True
    #                             )

    # test qa maxclique, sifferent kcore size, n10
    # for kcore in [1, 2, 3, 4]:
    #     qa_test_energies(f"benchmarkResults/energies/qa_energies_results_maxclique_kcore{kcore}_n10.json", 
    #                                 [{"n": 10, "p": 0.5, "k": None, "schedule": "linear_power", "B": 2, "T":20, "steps":30, "k_score_size": kcore}],
    #                                 generate_instance=generate_kcore_graph_instance,
    #                                 instance_count=150,
    #                                 problem=qa_max_clique_bqm,
    #                                 validate_solutions=helper_validate_max_clique_solutions,
    #                                 schedule_params=[1.2],
    #                                 iters=1,
    #                                 max_workers=10,
    #                                 append_graphs=True
    #                             )

    # for i in range(7):
    #     # test qa on maxclique, hard classical ER instances
    #     qa_test_energies(f"benchmarkResults/energies/qa_energies_results_maxclique_clas_hard_n10_p05_4.json", 
    #                                 [{"n": 10, "p": 0.5, "k": None, "schedule": "linear_power", "B": 2, "T":20, "steps":100}],
    #                                 generate_instance=lambda s: generate_random_instance_from_file("benchmark_data/er_hard_classical/max_clique/er_10_05"),
    #                                 instance_count=30,
    #                                 problem=qa_max_clique_bqm,
    #                                 validate_solutions=helper_validate_max_clique_solutions,
    #                                 schedule_params=[1.2],
    #                                 max_workers=10,
    #                                 append_graphs=True
    #                             )

    for i in range(30):
        print(f"Iteration {i}")
        # test qa on maxclique, different ER p
        qa_test_energies(f"benchmarkResults/energies/qa_energies_results_maxclique_er_p.json", 
                                    [{"n": 10, "p": float(p), "k": None, "schedule": "linear_power", "B": 2, "T":15, "steps":100} for p in np.linspace(0.01, 0.99, 50)],
                                    generate_instance=generate_random_graph_instance,
                                    instance_count=16,
                                    problem=qa_max_clique_bqm,
                                    validate_solutions=helper_validate_max_clique_solutions,
                                    schedule_params=[1.2],
                                    max_workers=8,
                                    append_graphs=True,
                                    eigenenergies=False
                                )

        # compare with regular ER instances
        # qa_test_energies(f"benchmarkResults/energies/qa_energies_results_maxclique_er_n10_p05_3.json", 
        #                             [{"n": 10, "p": 0.5, "k": None, "schedule": "linear_power", "B": 2, "T":20, "steps":100}],
        #                             generate_instance=generate_random_graph_instance,
        #                             instance_count=30,
        #                             problem=qa_max_clique_bqm,
        #                             validate_solutions=helper_validate_max_clique_solutions,
        #                             schedule_params=[1.2],
        #                             max_workers=10,
        #                             append_graphs=True
        #                         )

    # testing specific maxclique instance
    # qa_test_energies(f"benchmarkResults/energies/qa_energies_results_maxclique_clas_hard_n10_p05_one.json", 
    #                             [{"n": 10, "p": 0.5, "k": None, "schedule": "linear_power", "B": 2, "T":20, "steps":20}],
    #                             generate_instance=lambda s: generate_specific_instance_from_file("benchmark_data/er_hard_classical/max_clique/er_10_05", 15),
    #                             instance_count=1,
    #                             problem=qa_max_clique_bqm,
    #                             validate_solutions=helper_validate_max_clique_solutions,
    #                             schedule_params=[1.2],
    #                             iters=1,
    #                             max_workers=1,
    #                             append_graphs=True,
                                
    #                         )

    # max clique, different p
    # for i in range(5):
    #     for p in [9]:
    #         qa_test_energies(f"benchmarkResults/energies/qa_energies_results_maxclique_er_n10_p0{p}.json", 
    #                                 [{"n": 10, "p": p/10, "k": None, "schedule": "linear_power", "B": 2, "T":20, "steps":100}],
    #                                 generate_instance=generate_random_graph_instance,
    #                                 instance_count=30,
    #                                 problem=qa_max_clique_bqm,
    #                                 validate_solutions=helper_validate_max_clique_solutions,
    #                                 schedule_params=[1.2],
    #                                 max_workers=10,
    #                                 append_graphs=True
    #                             )

    # max clique, different n
    # for i in range(5):
    #     for n in [8, 10, 12]:
    #         qa_test_energies(f"benchmarkResults/energies/qa_energies_results_maxclique_er_n{n}_p06.json", 
    #                                 [{"n": n, "p": 0.6, "k": None, "schedule": "linear_power", "B": 2, "T":20, "steps":100}],
    #                                 generate_instance=generate_random_graph_instance,
    #                                 instance_count=10,
    #                                 problem=qa_max_clique_bqm,
    #                                 validate_solutions=helper_validate_max_clique_solutions,
    #                                 schedule_params=[1.2],
    #                                 max_workers=10,
    #                                 append_graphs=True
    #                             )