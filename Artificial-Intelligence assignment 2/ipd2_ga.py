import random
import matplotlib.pyplot as plt
import numpy as np
import time


# -------------------------------
# Genetic Strategy
# -------------------------------
class GeneticStrategy:
    def __init__(self, genome=None):
        # Genome: [initial_move, after_C, after_D]
        # Values are probabilities of cooperation (0 to 1)
        if genome is None:
            self.genome = [random.random() for _ in range(3)]
        else:
            self.genome = genome.copy()
        self.last_move = None
        self.last_opponent_move = None
        self.fitness = 0

    def make_move(self, opponent_last_move=None):
        if opponent_last_move is None:
            # First move
            move = "C" if random.random() < self.genome[0] else "D"
        else:
            # Subsequent moves based on opponent's last move
            if opponent_last_move == "C":
                move = "C" if random.random() < self.genome[1] else "D"
            else:  # opponent_last_move == "D"
                move = "C" if random.random() < self.genome[2] else "D"

        self.last_move = move
        self.last_opponent_move = opponent_last_move
        return move

    def reset(self):
        self.last_move = None
        self.last_opponent_move = None


# -------------------------------
# Game Logic
# -------------------------------
def play_game(strategy1, strategy2, iterations=200, noise_level=0.0):
    # Reset strategies
    strategy1.reset()
    strategy2.reset()

    # Payoff matrix (player1_score, player2_score)
    payoff_matrix = {
        ('C', 'C'): (3, 3),
        ('C', 'D'): (0, 5),
        ('D', 'C'): (5, 0),
        ('D', 'D'): (1, 1)
    }

    score1 = 0
    score2 = 0

    move1 = None
    move2 = None

    for _ in range(iterations):
        # Get intended moves
        move1 = strategy1.make_move(move2)
        move2 = strategy2.make_move(move1)

        # Apply noise if enabled
        if random.random() < noise_level:
            move1 = "D" if move1 == "C" else "C"
        if random.random() < noise_level:
            move2 = "D" if move2 == "C" else "C"

        round_score = payoff_matrix[(move1, move2)]
        score1 += round_score[0]
        score2 += round_score[1]

    return score1, score2


# -------------------------------
# Co-evolutionary Algorithm
# -------------------------------
def initialize_population(pop_size):
    return [GeneticStrategy() for _ in range(pop_size)]


def evaluate_population_coevolution(population, iterations=200, noise_level=0.0):
    # Reset fitness values
    for strategy in population:
        strategy.fitness = 0

    # Each strategy plays against each other strategy
    pop_size = len(population)
    for i in range(pop_size):
        for j in range(i + 1, pop_size):  # Don't play against self or duplicate games
            score_i, score_j = play_game(population[i], population[j], iterations, noise_level)
            population[i].fitness += score_i
            population[j].fitness += score_j

    # Average fitness by number of opponents
    for strategy in population:
        strategy.fitness /= (pop_size - 1)  # Played against every other member

    # Return list of fitnesses
    return [strategy.fitness for strategy in population]


def tournament_selection(population, fitnesses, k=3):
    selected = []
    pop_size = len(population)

    for _ in range(pop_size):
        # Select k random individuals
        competitors = random.sample(range(pop_size), k)

        # Find the one with the highest fitness
        winner = max(competitors, key=lambda i: fitnesses[i])

        # Add a copy to the selected individuals
        selected.append(GeneticStrategy(population[winner].genome))

    return selected


def crossover(parent1, parent2, crossover_rate=0.7):
    if random.random() > crossover_rate:
        return parent1, parent2

    # Single point crossover
    crossover_point = random.randint(1, len(parent1.genome) - 1)

    child1_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
    child2_genome = parent2.genome[:crossover_point] + parent1.genome[crossover_point:]

    return GeneticStrategy(child1_genome), GeneticStrategy(child2_genome)


def mutate(strategy, mutation_rate=0.1, mutation_step=0.1):
    for i in range(len(strategy.genome)):
        if random.random() < mutation_rate:
            # Add or subtract a small amount, keeping within [0, 1]
            delta = (random.random() * 2 - 1) * mutation_step
            strategy.genome[i] = max(0, min(1, strategy.genome[i] + delta))


def calculate_diversity(population):
    # Calculate population diversity as average pairwise distance between genomes
    pop_size = len(population)
    if pop_size <= 1:
        return 0

    total_distance = 0
    pairs = 0

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            # Calculate Euclidean distance between genomes
            dist = sum((population[i].genome[k] - population[j].genome[k]) ** 2 for k in
                       range(len(population[i].genome))) ** 0.5
            total_distance += dist
            pairs += 1

    return total_distance / pairs if pairs > 0 else 0


def run_coevolution(pop_size=50, generations=50, tournament_size=3, noise_level=0.0):
    # Initialize population
    population = initialize_population(pop_size)

    # Track statistics
    best_fitness_history = []
    avg_fitness_history = []
    diversity_history = []
    best_strategy = None
    best_fitness = -float('inf')

    # Evolution parameters
    crossover_rate = 0.7
    mutation_rate = 0.1

    # Start timing
    start_time = time.time()

    for generation in range(generations):
        # Evaluate fitness through co-evolution
        fitnesses = evaluate_population_coevolution(population, noise_level=noise_level)

        # Calculate diversity
        diversity = calculate_diversity(population)

        # Track statistics
        current_best_idx = fitnesses.index(max(fitnesses))
        current_best_fitness = fitnesses[current_best_idx]
        avg_fitness = sum(fitnesses) / len(fitnesses)

        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(avg_fitness)
        diversity_history.append(diversity)

        # Update overall best if better
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_strategy = GeneticStrategy(population[current_best_idx].genome)

        # Print progress every 10 generations
        if (generation + 1) % 10 == 0:
            print(
                f"Generation {generation + 1}: Best = {current_best_fitness:.2f}, Avg = {avg_fitness:.2f}, Diversity = {diversity:.4f}")

        # Selection
        selected = tournament_selection(population, fitnesses, tournament_size)

        # Create new population
        new_population = []

        while len(new_population) < pop_size:
            # Select two parents
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)

            # Crossover
            child1, child2 = crossover(parent1, parent2, crossover_rate)

            # Mutation
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)

            # Add to new population
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        # Replace old population
        population = new_population

    # Calculate runtime
    runtime = time.time() - start_time

    # Return results
    return {
        "best_strategy": best_strategy,
        "best_fitness": best_fitness,
        "best_fitness_history": best_fitness_history,
        "avg_fitness_history": avg_fitness_history,
        "diversity_history": diversity_history,
        "runtime": runtime,
        "final_population": population
    }


def interpret_strategy(strategy):
    # Classify the strategy based on its genome
    interp = {
        "initial_move": "Cooperate" if strategy.genome[0] > 0.5 else "Defect",
        "after_C": f"{strategy.genome[1]:.2f} probability of cooperation",
        "after_D": f"{strategy.genome[2]:.2f} probability of cooperation"
    }

    # Simple classification
    if strategy.genome[1] > 0.8 and strategy.genome[2] < 0.2:
        interp["classification"] = "Similar to Tit-for-Tat"
    elif strategy.genome[1] > 0.8 and strategy.genome[2] > 0.8:
        interp["classification"] = "Similar to Always Cooperate"
    elif strategy.genome[1] < 0.2 and strategy.genome[2] < 0.2:
        interp["classification"] = "Similar to Always Defect"
    else:
        interp["classification"] = "Mixed Strategy"

    return interp


def plot_results(results):
    # Plot fitness and diversity
    plt.figure(figsize=(12, 8))

    # Plot fitness
    plt.subplot(2, 1, 1)
    plt.plot(results["best_fitness_history"], label='Best Fitness')
    plt.plot(results["avg_fitness_history"], label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True)

    # Plot diversity
    plt.subplot(2, 1, 2)
    plt.plot(results["diversity_history"], label='Population Diversity', color='green')
    plt.xlabel('Generation')
    plt.ylabel('Diversity')
    plt.title('Population Diversity')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('coevolution_results.png')
    plt.show()

    # Plot strategy distribution
    final_population = results["final_population"]
    after_C_values = [s.genome[1] for s in final_population]
    after_D_values = [s.genome[2] for s in final_population]
    initial_move_values = [s.genome[0] for s in final_population]

    plt.figure(figsize=(10, 8))
    plt.scatter(after_C_values, after_D_values, c=initial_move_values, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Initial Move (probability of C)')
    plt.xlabel('Probability of C after C')
    plt.ylabel('Probability of C after D')
    plt.title('Final Population Strategy Distribution')




# -------------------------------
# Main Function
# -------------------------------
def main():
    print("Starting Co-evolutionary Algorithm for IPD - Part 2")

    # Run co-evolution
    results = run_coevolution(pop_size=50, generations=50)

    # Print results
    print("\nCo-evolution Results:")
    print(f"Runtime: {results['runtime']:.2f} seconds")
    print(f"Best Fitness: {results['best_fitness']}")
    print(f"Best Strategy Genome: {[f'{g:.2f}' for g in results['best_strategy'].genome]}")

    # Interpret best strategy
    interpretation = interpret_strategy(results['best_strategy'])
    print("\nBest Strategy Interpretation:")
    for key, value in interpretation.items():
        print(f"  {key}: {value}")

    # Plot results
    plot_results(results)



if __name__ == "__main__":
    main()