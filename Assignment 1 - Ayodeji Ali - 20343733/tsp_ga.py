import random
import math
import time
import numpy as np


# -------------------------------
# TSPLIB Parser and Distance Matrix (using NumPy)
# -------------------------------
def read_tsplib(filename):
    """
    Reads a TSPLIB file and returns a list of (x, y) coordinates.
    """
    coords = []
    with open(filename) as f:
        # Skip lines until we find the NODE_COORD_SECTION
        for line in f:
            if line.strip().upper().startswith("NODE_COORD_SECTION"):
                break
        # Read coordinate data until "EOF" or an empty line is encountered
        for line in f:
            line = line.strip()
            if line == "EOF" or not line:
                break
            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                coords.append((x, y))
    return coords


def create_distance_matrix(coords):
    """
    Creates a distance matrix using NumPy vectorized operations.
    """
    arr = np.array(coords)  # shape: (n, 2)
    # Compute pairwise differences using broadcasting
    diff = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
    # Compute Euclidean distances
    dist_matrix = np.hypot(diff[:, :, 0], diff[:, :, 1])
    return dist_matrix


# -------------------------------
# Genetic Algorithm Components
# -------------------------------
def initialize_population(pop_size, num_cities):
    """
    Initializes a population of random tours (each tour is a permutation of city indices).
    """
    base = list(range(num_cities))
    return [random.sample(base, len(base)) for _ in range(pop_size)]


def tour_length(tour, dist_matrix):
    """
    Calculates the total tour length using NumPy vectorized indexing.
    """
    tour_arr = np.array(tour)
    next_arr = np.roll(tour_arr, -1)  # shift left to get next city (wraps around)
    return dist_matrix[tour_arr, next_arr].sum()


def evaluate_population(population, dist_matrix):
    """
    Computes the fitness for each individual in the population.
    Fitness is defined as the inverse of the tour length.
    """
    return [1.0 / tour_length(ind, dist_matrix) for ind in population]


def tournament_selection(population, fitnesses, tournament_size=5):
    """
    Selects individuals from the population using tournament selection.
    """
    n = len(population)
    selected = []
    for _ in range(n):
        # Pick a random subset of individuals
        tournament = random.sample(range(n), tournament_size)
        # Choose the individual with the highest fitness
        best = max(tournament, key=lambda i: fitnesses[i])
        selected.append(population[best][:])
    return selected


# -------------------------------
# Crossover Operators
# -------------------------------
def order_crossover(parent1, parent2):
    """
    Performs Order Crossover (OX) on two parents.
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    # Copy a slice from parent1
    child[a:b + 1] = parent1[a:b + 1]
    pos = (b + 1) % size
    # Fill in remaining positions with genes from parent2 in order
    for i in range(size):
        gene = parent2[(b + 1 + i) % size]
        if gene not in child:
            child[pos] = gene
            pos = (pos + 1) % size
    return child


def pmx_crossover(parent1, parent2):
    """
    Performs Partially Mapped Crossover (PMX) on two parents.
    """
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b + 1] = parent1[a:b + 1]
    for i in range(a, b + 1):
        gene = parent2[i]
        if gene not in child:
            pos = i
            while True:
                pos = parent2.index(parent1[pos])
                if child[pos] is None:
                    child[pos] = gene
                    break
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]
    return child


# -------------------------------
# Mutation Operators
# -------------------------------
def swap_mutation(individual):
    """
    Performs swap mutation by exchanging two cities in the tour.
    """
    a, b = random.sample(range(len(individual)), 2)
    individual[a], individual[b] = individual[b], individual[a]


def inversion_mutation(individual):
    """
    Performs inversion mutation by reversing the order of cities between two positions.
    """
    a, b = sorted(random.sample(range(len(individual)), 2))
    individual[a:b + 1] = individual[a:b + 1][::-1]


# -------------------------------
# Main GA Function
# -------------------------------
def run_ga(dist_matrix, pop_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.2):
    """
    Runs the Genetic Algorithm for the TSP.
    Returns the best solution, its tour length, the fitness history, and computational time.
    """
    num_cities = dist_matrix.shape[0]
    population = initialize_population(pop_size, num_cities)
    best_solution, best_length = None, float('inf')
    fitness_history = []
    start_time = time.time()

    for _ in range(generations):
        fitnesses = evaluate_population(population, dist_matrix)
        for ind in population:
            L = tour_length(ind, dist_matrix)
            if L < best_length:
                best_length, best_solution = L, ind[:]
        fitness_history.append(1.0 / best_length)

        selected = tournament_selection(population, fitnesses, tournament_size=5)
        new_population = []
        while len(new_population) < pop_size:
            p1, p2 = random.choice(selected), random.choice(selected)
            if random.random() < crossover_rate:
                # Randomly choose between Order Crossover and PMX
                child = order_crossover(p1, p2) if random.random() < 0.5 else pmx_crossover(p1, p2)
            else:
                child = p1[:]
            if random.random() < mutation_rate:
                # Randomly choose between swap and inversion mutation
                swap_mutation(child) if random.random() < 0.5 else inversion_mutation(child)
            new_population.append(child)
        population = new_population

    comp_time = time.time() - start_time
    return best_solution, best_length, fitness_history, comp_time



if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Usage: python tsp_ga.py <tsplib_file>")
        sys.exit(1)
    coords = read_tsplib(sys.argv[1])
    if not coords:
        print("Error reading file.")
        sys.exit(1)
    dist_matrix = create_distance_matrix(coords)
    best, best_len, history, comp_time = run_ga(dist_matrix)
    print("Best tour:", best)
    print("Tour length:", best_len)
    print("Computational time:", comp_time)
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (1/Tour Length)")
    plt.title("Fitness Evolution")
    plt.show()
