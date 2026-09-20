from benchmark import qa_bo_optimize_schedule
from annealing import qa_k_clique_bqm, qa_max_clique_bqm
from helper_functions import helper_validate_k_clique_solutions, helper_validate_max_clique_solutions
from utils import generate_k_clique_instance, generate_random_graph_instance

if __name__ == "__main__":

    # optimize qa schedule function using BO
    qa_bo_optimize_schedule("benchmarkResults/schedule_optimization/optimizer_state_maxclique_lininterp_n8_p6.json", 
                            [{"n": 8, "p": 0.6, "k": None, "schedule_function": "lin_interp"}],
                            generate_instance=generate_random_graph_instance,
                            instance_count=300,
                            problem=qa_max_clique_bqm,
                            validate_solutions=helper_validate_max_clique_solutions,
                            iters=1,
                            max_workers=12,
                            return_eigenenergies=False,
                            n_iter=50,
                            n_params=5,
                            probe_points=[[1/6, 1/6, 1/6, 1/6, 1/6]]
                        )