from benchmark import qa_bo_optimize_schedule
from annealing import qa_k_clique_bqm, qa_max_clique_bqm
from helper_functions import helper_validate_k_clique_solutions, helper_validate_max_clique_solutions
from utils import generate_k_clique_instance, generate_random_graph_instance

if __name__ == "__main__":

    # optimize qa schedule function using BO
    qa_bo_optimize_schedule("benchmarkResults/schedule_optimization/optimizer_state_maxclique_linearpower_n10_p4.json", 
                            [{"n": 10, "p": 0.4, "k": None, "schedule": "linear_power", "B": 4, "T":20}],
                            generate_instance=generate_random_graph_instance,
                            instance_count=50,
                            problem=qa_max_clique_bqm,
                            validate_solutions=helper_validate_max_clique_solutions,
                            iters=1,
                            max_workers=10,
                            return_eigenenergies=False,
                            n_iter=50,
                            n_params=1,
                            probe_points=[[1]],
                            alpha=0.04,
                            bounds=(0.7, 1.3),
                            parameter_type="absolute"
                        )
