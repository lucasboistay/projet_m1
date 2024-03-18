# Description: This file contains the constants used in the simulated annealing algorithm.

# General model constants

N = 100 # Number of rows
M = 100 # Number of columns
iterations = 1000000 # Number of iterations
J_values = [0.2, 0.5, 0.7, 1.3, 1.5, 2.] # Interaction constant list

# Parallel processing constants

number_of_simulations = 1000 # Number of simulations for the critical temperature computation
number_of_pool_processes = 8  # Number of pool processes, do not set it to more than the number of cores of your CPU

# Temperature constants

t_min = 0.1 # Minimum temperature for the critical temperature computation
t_max = 6 # Maximum temperature for the critical temperature computation
