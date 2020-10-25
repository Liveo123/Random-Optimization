import six
import sys

sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# import sys
# import itertools

# Dictionaries
RHC_KEY = 'rhc'
SA_KEY = 'sa'
GA_KEY = 'ga'
MIMIC_KEY = 'mimic'

# Algorithm Names
RHC_ALGORITHM = 'Random Hill Climb'
SA_ALGORITHM = 'Simulated Annealing'
GA_ALGORITHM = 'Genetic Algormithm'
MIMIC_ALGORITHM = 'MIMIC'

# Different charts
FITNESS_PLOT = 0
TIME_PLOT = 1

# Chart axes
ITERATIONS_AXIS = 'Iterations'
TIME_AXIS = 'Time'
FITNESS_AXIS = 'Fitness'
ALGORITHM_AXIS = 'Algorithm'

# Tuning
MAX_ITEMS = 50
MAX_WEIGHTS = 40
MAX_VALUES = 5
NUM_OF_EACH = 30
RANDOM_SEED = 10
MAX_WEIGHT_PCT = 0.6
MAX_VAL = 3
SA = 0.9
RESTARTS = 300
MAX_ATTEMPTS = 200
MAX_ITERS = 100000
AVG_ITERS = 10
POPULATION_SIZE = 100
MUTATION_PROB = 0.1
KEEP_PCT = 0.2

types_of_items = 35
maximum_items = 20

# Weights of each of the possible items to add
weights = np.random.randint(1, MAX_WEIGHTS, size=NUM_OF_EACH)
# Values of each of the possible items to add
values = np.random.randint(1, MAX_VALUES, size=NUM_OF_EACH)
# MAX_WEIGHT_PCT is the percentage of knapack weight used for each item type
fitness = mlrose.Knapsack(weights.tolist(), values.tolist(), MAX_WEIGHT_PCT)
print(fitness.values)
print(fitness.weights)
print(fitness.get_prob_type())

# length is number of elements in the state vector (int)
# fitness function is the value returned by mlrose.knapsack
# Is this a maximize or minimize problem
# max_cal = max value of each element in state vector - i.e. max number of each item type
prob = mlrose.DiscreteOpt(length=NUM_OF_EACH,
                          fitness_fn=fitness,
                          maximize=True,
                          max_val=MAX_VAL)
# initial_state = np.random.randint(MAX_VAL+1, size=NUM_OF_EACH)
initial_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0]
schedule = mlrose.GeomDecay(1000, 0.9, 1)

# prob - Object containing fitness function optimization problem to be solved.
# For example, DiscreteOpt(), ContinuousOpt() or TSPOpt()
# max_attempts - Maximum number of attempts to find a better neighbor at each step
# restarts - Number of random restarts
# max_iters - Maximum number of iterations of the algorithm.
# init_state - 1-D Numpy array containing starting state for algorithm. If None, then a
# random state is used.

# iterations = 5000
# start_time = time.time()
# rnd_initial_state = np.array(initial_state)
# best_rhc_state, best_rhc_fitness = mlrose.random_hill_climb(prob,
#                                                             max_attempts=MAX_ATTEMPTS,
#                                                             restarts=RESTARTS,
#                                                             max_iters=iterations,
#                                                             init_state=rnd_initial_state,
#                                                             random_state = RANDOM_SEED)
#
# best_sa_state, best_sa_fitness = mlrose.simulated_annealing(prob,
#                                                             max_attempts=MAX_ATTEMPTS,
#                                                             schedule=schedule,
#                                                             max_iters=MAX_ITERS,
#                                                             init_state=rnd_initial_state)
#
# print(f'time = {time.time() - start_time}')
# print(f'best rhc state = {best_rhc_state}')
# print(f'fitness rhc evaluation = {fitness.evaluate(best_rhc_state)}')
# print(f'best sa state = {best_sa_state}'['Iterations', 'Fitness')
# print(f'fitness sa evaluation = {fitness.evaluate(best_sa_state)}')

# sys.exit()

# Run the experiment
fitness_results = {RHC_KEY: [], SA_KEY: [], GA_KEY: [], MIMIC_KEY: []}
rnd_initial_state = np.array(initial_state)
best_fitness = {RHC_KEY: 0, SA_KEY: 0, GA_KEY: 0, MIMIC_KEY: 0}
best_state = {RHC_KEY: 0, SA_KEY: 0, GA_KEY: 0, MIMIC_KEY: 0}

time_rhc = 0
time_sa = 0
time_ga = 0
time_mimic = 0

for iterations in [1,
                   3]:  # , 10, 20, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900,
    # , 1000, 3000, 5000, 10000, 50000, 75000, 100000, 150000, 200000, 250000]: #range(1, MAX_ITERS, 100):
    # for iterations in [1, 3, 5, 10, 20, 50, 75, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
    # , 1000, 3000, 5000, 10000, 50000, 75000, 100000, 150000, 200000, 250000]: #range(1, MAX_ITERS, 100):
    temp_fitness = {RHC_KEY: 0, SA_KEY: 0, GA_KEY: 0, MIMIC_KEY: 0}
    for rnd_iterations in range(0, AVG_ITERS):
        # Random Hill Climb
        start_time = time.time()
        best_state = {}

        best_state[RHC_KEY], best_fitness[RHC_KEY] = mlrose.random_hill_climb(prob,
                                                                              max_attempts=MAX_ATTEMPTS,
                                                                              restarts=RESTARTS,
                                                                              max_iters=iterations,
                                                                              init_state=rnd_initial_state,
                                                                              random_state=RANDOM_SEED)
        time_rhc = time.time() - start_time
        temp_fitness[RHC_KEY] += best_fitness[RHC_KEY]

        # Simulated Annealing
        start_time = time.time()
        best_state[SA_KEY], best_fitness[SA_KEY] = mlrose.simulated_annealing(prob,
                                                                              max_attempts=MAX_ATTEMPTS,
                                                                              schedule=schedule,
                                                                              max_iters=MAX_ITERS,
                                                                              init_state=rnd_initial_state)
        time_sa = time.time() - start_time
        temp_fitness[SA_KEY] += best_fitness[SA_KEY]

        # Genetic Algorithm
        start_time = time.time()
        best_state[GA_KEY], best_fitness[GA_KEY] = mlrose.genetic_alg(prob,
                                                                      max_attempts=MAX_ATTEMPTS,
                                                                      # schedule=schedule,
                                                                      max_iters=MAX_ITERS,
                                                                      mutation_prob=MUTATION_PROB,
                                                                      pop_size=POPULATION_SIZE)
        time_ga = time.time() - start_time
        temp_fitness[GA_KEY] += best_fitness[GA_KEY]

        # MIMIC
        start_time = time.time()
        best_state[MIMIC_KEY], best_fitness[MIMIC_KEY] = mlrose.mimic(prob,
                                                                      max_attempts=MAX_ATTEMPTS,
                                                                      # schexdule=schedule,
                                                                      max_iters=MAX_ITERS,
                                                                      # init_state=rnd_initial_state,
                                                                      pop_size=POPULATION_SIZE,
                                                                      keep_pct=KEEP_PCT,
                                                                      fast_mimic=True)
        time_mimic = time.time() - start_time
        temp_fitness[MIMIC_KEY] += best_fitness[MIMIC_KEY]

    fitness_results[RHC_KEY].append(
        [iterations, temp_fitness[RHC_KEY] / AVG_ITERS, time_rhc])
    print(f'Random Hill Climb finished for {iterations} iterations.')
    fitness_results[SA_KEY].append(
        [iterations, temp_fitness[SA_KEY] / AVG_ITERS, time_sa])
    print(f'Simulated Annealing finished for {iterations} iterations.')
    fitness_results[GA_KEY].append(
        [iterations, temp_fitness[GA_KEY] / AVG_ITERS, time_ga])
    print(f'Genetic Algorithm finished for {iterations} iterations.')
    fitness_results[MIMIC_KEY].append(
        [iterations, temp_fitness[MIMIC_KEY] / AVG_ITERS, time_mimic])
    print(f'MIMIC finished for {iterations} iterations.')

# Create multiple plots at once
fig, ax = plt.subplot(1, 2)

# Build RHC line plot for Fitness for Iteration count
fitness_results[RHC_KEY] = pd.DataFrame(fitness_results[RHC_KEY],
                                        columns=[ITERATIONS_AXIS,
                                                 FITNESS_AXIS,
                                                 TIME_AXIS])
pd[ALGORITHM_AXIS] = RHC_ALGORITHM * fitness_results[RHC_KEY].count()
sns.lineplot(data=fitness_results[RHC_KEY],
             x=ITERATIONS_AXIS,
             y=FITNESS_AXIS,
             marker='o',
             legend=True,
             hue=ALGORITHM_AXIS,
             ax=ax[FITNESS_PLOT])

# Build RHC line plot for Time for Iteration count
# sns.lineplot(data=fitness_results[RHC_KEY],
#              x=ITERATIONS_AXIS,
#              y=TIME_AXIS,
#              marker='o',
#              legend=True,
#              hue=ALGORITHM_AXIS,
#              ax=ax[TIME_PLOT])

# Build SA line plot for Fitness for Iteration count
fitness_results[SA_KEY] = pd.DataFrame(fitness_results[SA_KEY], columns=[ITERATIONS_AXIS,
                                                                         FITNESS_AXIS,
                                                                         TIME_AXIS])
pd[ALGORITHM_AXIS] = SA_ALGORITHM * fitness_results[SA_KEY].count()
sns.lineplot(data=fitness_results[SA_KEY],
             x=ITERATIONS_AXIS,
             y=FITNESS_AXIS,
             marker='o',
             legend=True,
             hue=ALGORITHM_AXIS,
             ax=ax[FITNESS_PLOT])

# Build GA line plot for Fitness for Iteration count
fitness_results[GA_KEY] = pd.DataFrame(fitness_results[GA_KEY], columns=[ITERATIONS_AXIS,
                                                                         FITNESS_AXIS,
                                                                         TIME_AXIS])
pd[ALGORITHM_AXIS] = GA_ALGORITHM * fitness_results[GA_KEY].count()
sns.lineplot(data=fitness_results[GA_KEY],
             x=ITERATIONS_AXIS,
             y=FITNESS_AXIS,
             marker='o',
             legend=True,
             hue=ALGORITHM_AXIS,
             ax=ax[FITNESS_PLOT])

# Build MIMIC line plot for Fitness for Iteration count
fitness_results[MIMIC_KEY] = pd.DataFrame(fitness_results[MIMIC_KEY],
                                          columns=[ITERATIONS_AXIS,
                                                   FITNESS_AXIS,
                                                   TIME_AXIS])
pd[ALGORITHM_AXIS] = MIMIC_ALGORITHM * fitness_results[MIMIC_KEY].count()
sns.lineplot(data=fitness_results[MIMIC_KEY],
             x=ITERATIONS_AXIS,
             y=FITNESS_AXIS,
             marker='o',
             legend=True,
             hue=ALGORITHM_AXIS,
             ax=ax[FITNESS_PLOT])

# Create grid background.
sns.set_style("darkgrid", {'axes.axisbelow': True})
fig.show()
