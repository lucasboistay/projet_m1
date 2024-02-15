# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 17:27:40 2024

@author: roron, lucas
"""
import random
import matplotlib.pyplot as plt
import numpy as np

def map_rw(n: int, x0: float):
    """
    Simulates one random walk on the line.

    :param n: Number of iterations.
    :type n: int
    :param x0: Initial position.
    :type x0: float
    :return: List of positions after each iteration.
    :rtype: list
    """
    positionL = [x0]
    position = x0
    for i in range(n):
        step = random.choice([-1, 1])
        positionL.append(positionL[i] + step)
        position += step
    return positionL

def plot_random_walks(n_iterations: int, nb_run: int):
    """
    Plots random walks for a given number of iterations and runs.

    :param n_iterations: Number of iterations.
    :type n_iterations: int
    :param nb_run: Number of runs.
    :type nb_run: int
    """
    print("Plotting random walks...")
    Lx = list(range(n_iterations + 1))
    List_df = []  # List of final destinations

    for k in range(nb_run):
        y = map_rw(n_iterations, 0)
        plt.plot(Lx, y)
        List_df.append(y[n_iterations])

    plt.xlabel('Number of steps')
    plt.ylabel('Position')
    plt.savefig('data/random_walks.png', dpi=300)
    plt.show()
    print("Mean final position:", np.mean(List_df))
    print("Random walks saved as random_walks.png")

def plot_mean_initial_positions(n_iterations: int, nb_run: int):
    """
    Plots the mean final position for different initial positions.

    :param n_iterations: Number of iterations.
    :type n_iterations: int
    :param nb_run: Number of runs.
    :type nb_run: int
    """
    print("Plotting mean final position...")
    L_mean = []
    comptage = []

    for p in range(20):
        L = []
        comptage.append(p)
        for x in range(nb_run):
            y = map_rw(n_iterations, p)
            L.append(y[n_iterations])
        L_mean.append(np.mean(L))

    plt.plot(comptage, L_mean)
    plt.xlabel('Initial Position')
    plt.ylabel('Mean Final Position')
    plt.savefig('data/mean_final_position.png', dpi=300)
    plt.show()
    print("Mean final position saved as mean_final_position.png")

def plot_histogram(List_df: list):
    """
    Plots a histogram of the final positions.

    :param List_df: List of final positions.
    :type List_df: list
    """
    print("Plotting histogram...")
    plt.hist(List_df, range=(-60, 60), bins=120)
    plt.xlabel('Number of occurrences')
    plt.ylabel('Final Positions')
    plt.title('Example of a simple histogram')
    plt.savefig('data/histogram.png', dpi=300)
    plt.show()
    print("Histogram saved as histogram.png")

def plot_variance_over_time(n_iterations: int, nb_run: int):
    """
    Plots the variance of positions over time.

    :param n_iterations: Number of iterations.
    :type n_iterations: int
    :param nb_run: Number of runs.
    :type nb_run: int
    """
    Var = []
    comptage = []

    print("Computing variance over time...")

    for t in range(n_iterations):
        L = []
        comptage.append(t)
        for x in range(nb_run):
            y = map_rw(t, x0)
            L.append((y[t] - x0) ** 2)
        Var.append(np.mean(L))

    plt.plot(comptage, Var)
    plt.xlabel('Elapsed time')
    plt.ylabel('Variance')
    plt.savefig('data/variance.png', dpi=300)
    plt.show()
    print("Variance over time saved as variance.png")

# Example usage:
n_iterations = 200
nb_run = 20000
x0 = 0

plot_random_walks(n_iterations, nb_run)
plot_mean_initial_positions(n_iterations, nb_run)

List_df = [map_rw(n_iterations, 0)[-1] for _ in range(nb_run)]
plot_histogram(List_df)

plot_variance_over_time(n_iterations, nb_run)