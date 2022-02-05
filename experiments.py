from ioh import get_problem
from ioh import logger
from genetic_algorithm import GeneticAlgorithm

# Declaration of problems to be tested.
om = get_problem(1, dim=50, iid=1, problem_type = "PBO")
lo = get_problem(2, dim=50, iid=1, problem_type = "PBO")
labs = get_problem(18, dim=32, iid=1, problem_type = "PBO")

""" OneMax """

print("\nOneMax experiment 1")
l = logger.Analyzer(root="data", 
    folder_name="run_om_1",
    algorithm_name="ga_om_exp1", 
    algorithm_info="Genetic Algorithm for One Max problem exp.1")

om.attach_logger(l)
om_ga = GeneticAlgorithm(om, initial_pop=10, plus=False)
om_ga(mating="tournament", mating_params=[10, 10, 0.6], crossover="n-points", cross_params=[7], p_m=1/50, seed=55)

print("\nOneMax experiment 2")
l = logger.Analyzer(root="data", 
    folder_name="run_om_2",
    algorithm_name="ga_om_exp2",
    algorithm_info="Genetic Algorithm for One Max problem exp.2")

om.attach_logger(l)
om_ga = GeneticAlgorithm(om, initial_pop=50, plus=False)
om_ga(mating="tournament", mating_params=[50, 10, 0.6], crossover="n-points", cross_params=[7], p_m=1/50, seed=55)

print("\nOneMax experiment 3")
l = logger.Analyzer(root="data", 
    folder_name="run_om_3",
    algorithm_name="ga_om_exp3", 
    algorithm_info="Genetic Algorithm for One Max problem exp.3")

om.attach_logger(l)
om_ga = GeneticAlgorithm(om, initial_pop=10, plus=True)
om_ga(mating="tournament", mating_params=[10, 10, 0.6], crossover="n-points", cross_params=[7], p_m=None, seed=55)

print("\nOneMax experiment 4")
l = logger.Analyzer(root="data", 
    folder_name="run_om_4",
    algorithm_name="ga_om_exp4", 
    algorithm_info="Genetic Algorithm for One Max problem exp.4")

om.attach_logger(l)
om_ga = GeneticAlgorithm(om, initial_pop=10, plus=False)
om_ga(mating="tournament", mating_params=[10, 10, 0.6], crossover="uniform", cross_params=[], p_m=1/50, seed=55)

print("\nOneMax experiment 5")
l = logger.Analyzer(root="data", 
    folder_name="run_om_5",
    algorithm_name="ga_om_exp5", 
    algorithm_info="Genetic Algorithm for One Max problem exp.5")

om.attach_logger(l)
om_ga = GeneticAlgorithm(om, initial_pop=10, plus=True)
om_ga(mating="proportionate", mating_params=[10], crossover="n-points", cross_params=[7], p_m=None, seed=55)

""" LeadingOnes """

print("\nLeadingOnes experiment 1")
l = logger.Analyzer(root="data", 
    folder_name="run_lo_1",
    algorithm_name="ga_lo_exp1", 
    algorithm_info="Genetic Algorithm for Leading Ones problem exp.1")

lo.attach_logger(l)
lo_ga = GeneticAlgorithm(lo, initial_pop=30, plus=True)
lo_ga(mating="tournament", mating_params=[10, 10, 0.6], crossover="n-points", cross_params=[4], p_m=1/50, seed=80)

print("\nLeadingOnes experiment 2")
l = logger.Analyzer(root="data", 
    folder_name="run_lo_2",
    algorithm_name="ga_lo_exp2", 
    algorithm_info="Genetic Algorithm for Leading Ones problem exp.2")

lo.attach_logger(l)
lo_ga = GeneticAlgorithm(lo, initial_pop=30, plus=False)
lo_ga(mating="tournament", mating_params=[10, 10, 0.6], crossover="n-points", cross_params=[4], p_m=1/50, seed=80)

print("\nLeadingOnes experiment 3")
l = logger.Analyzer(root="data", 
    folder_name="run_lo_3",
    algorithm_name="ga_lo_exp3",
    algorithm_info="Genetic Algorithm for Leading Ones problem exp.3")

lo.attach_logger(l)
lo_ga = GeneticAlgorithm(lo, initial_pop=30, plus=False)
lo_ga(mating="tournament", mating_params=[10, 10, 0.6], crossover="uniform", cross_params=[], p_m=1/50, seed=80)

print("\nLeadingOnes experiment 4")
l = logger.Analyzer(root="data", 
    folder_name="run_lo_4",
    algorithm_name="ga_lo_exp4",
    algorithm_info="Genetic Algorithm for Leading Ones problem exp.4")

lo.attach_logger(l)
lo_ga = GeneticAlgorithm(lo, initial_pop=30, plus=False)
lo_ga(mating="tournament", mating_params=[30, 30, 0.6], crossover="uniform", cross_params=[], p_m=1/25, seed=80)

""" LABS """

print("\nLABS experiment 1")
l = logger.Analyzer(root="data", 
    folder_name="run_labs_1",
    algorithm_name="ga_labs_exp1", 
    algorithm_info="Genetic Algorithm for LABS problem exp.1")

labs.attach_logger(l)
labs_ga = GeneticAlgorithm(labs, initial_pop=10, plus=False)
labs_ga(mating="proportionate", mating_params=[10], crossover="n-points", cross_params=[4], p_m=1/10, seed=55)

print("\nLABS experiment 2")
l = logger.Analyzer(root="data", 
    folder_name="run_labs_2",
    algorithm_name="ga_labs_exp2", 
    algorithm_info="Genetic Algorithm for LABS problem exp.2")

labs.attach_logger(l)
labs_ga = GeneticAlgorithm(labs, initial_pop=10, plus=True)
labs_ga(mating="proportionate", mating_params=[10], crossover="n-points", cross_params=[4], p_m=1/10, seed=55)

print("\nLABS experiment 3")
l = logger.Analyzer(root="data", 
    folder_name="run_labs_3",
    algorithm_name="ga_labs_exp3", 
    algorithm_info="Genetic Algorithm for LABS problem exp.3")

labs.attach_logger(l)
labs_ga = GeneticAlgorithm(labs, initial_pop=10, plus=True)
labs_ga(mating="tournament", mating_params=[10,4,0.6], crossover="n-points", cross_params=[4], p_m=1/10, seed=55)

print("\nLABS experiment 4")
l = logger.Analyzer(root="data",
    folder_name="run_labs_4",
    algorithm_name="ga_labs_exp4",
    algorithm_info="Genetic Algorithm for LABS problem exp.4")

labs.attach_logger(l)
labs_ga = GeneticAlgorithm(labs, initial_pop=10, plus=True)
labs_ga(mating="tournament", mating_params=[10,4,0.6], crossover="uniform", cross_params=[], p_m=1/10, seed=55)