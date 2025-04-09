import random
import matplotlib.pyplot as plt
import numpy as np
import time


# -------------------------------
# Fixed Strategies
# -------------------------------
class FixedStrategy:
    def __init__(self, name):
        self.name = name
        self.last_move = None

    def reset(self):
        self.last_move = None


class AlwaysCooperate(FixedStrategy):
    def __init__(self):
        super().__init__("Always Cooperate")

    def make_move(self, opponent_last_move=None):
        self.last_move = "C"
        return "C"


class AlwaysDefect(FixedStrategy):
    def __init__(self):
        super().__init__("Always Defect")

    def make_move(self, opponent_last_move=None):
        self.last_move = "D"
        return "D"


class TitForTat(FixedStrategy):
    def __init__(self):
        super().__init__("Tit For Tat")

    def make_move(self, opponent_last_move=None):
        if opponent_last_move is None:
            self.last_move = "C"
            return "C"
        self.last_move = opponent_last_move
        return opponent_last_move


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
def play_game(strategy1, strategy2, iterations=200):
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
        move1 = strategy1.make_move(move2)
        move2 = strategy2.make_move(move1)

        round_score = payoff_matrix[(move1, move2)]
        score1 += round_score[0]
        score2 += round_score[1]

    return score1, score2


# -------------------------------
# Genetic Algorithm
# -------------------------------
def initialize_population(pop_size):
    return [GeneticStrategy() for _ in range(pop_size)]


def evaluate_fitness(population, fixed_strategies, iterations=200):
    fitnesses = []

    for strategy in population:
        total_score = 0
        for opponent in fixed_strategies:
            score, _ = play_game(strategy, opponent, iterations)
            total_score += score
        fitnesses.append(total_score)

    return fitnesses


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


def run_genetic_algorithm(pop_size=50, generations=50, tournament_size=3):
    # Create fixed strategies
    fixed_strategies = [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat()
    ]

    # Initialize population
    population = initialize_population(pop_size)

    # Track best strategy and fitness
    best_fitness_history = []
    avg_fitness_history = []
    best_strategy = None
    best_fitness = -float('inf')

    # Evolution parameters
    crossover_rate = 0.7
    mutation_rate = 0.1

    # Start timing
    start_time = time.time()

    for generation in range(generations):
        # Evaluate fitness
        fitnesses = evaluate_fitness(population, fixed_strategies)

        # Track statistics
        current_best_idx = fitnesses.index(max(fitnesses))
        current_best_fitness = fitnesses[current_best_idx]
        avg_fitness = sum(fitnesses) / len(fitnesses)

        best_fitness_history.append(current_best_fitness)
        avg_fitness_history.append(avg_fitness)

        # Update overall best if better
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_strategy = GeneticStrategy(population[current_best_idx].genome)

        # Print progress every 10 generations
        if (generation + 1) % 10 == 0:
            print(
                f"Generation {generation + 1}: Best Fitness = {current_best_fitness}, Avg Fitness = {avg_fitness:.2f}")

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

    # Analyze best strategy
    analysis = analyze_best_strategy(best_strategy, fixed_strategies)

    # Return results
    return {
        "best_strategy": best_strategy,
        "best_fitness": best_fitness,
        "best_fitness_history": best_fitness_history,
        "avg_fitness_history": avg_fitness_history,
        "runtime": runtime,
        "analysis": analysis
    }


def analyze_best_strategy(strategy, fixed_strategies):
    results = {}

    # Interpret the genome
    interpretation = {
        "initial_move": "Cooperate" if strategy.genome[0] > 0.5 else "Defect",
        "after_C": f"{strategy.genome[1]:.2f} probability of cooperation",
        "after_D": f"{strategy.genome[2]:.2f} probability of cooperation"
    }

    # Test against fixed strategies
    for opponent in fixed_strategies:
        score, opponent_score = play_game(strategy, opponent)
        results[opponent.name] = {
            "strategy_score": score,
            "opponent_score": opponent_score
        }

    return {
        "interpretation": interpretation,
        "performance": results
    }


def plot_fitness_history(best_fitness_history, avg_fitness_history):
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progression Over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig('fitness_history.png')
    plt.show()


# -------------------------------
# Main Function
# -------------------------------
def main():
    print("Starting Genetic Algorithm for IPD - Part 1")

    # Run the genetic algorithm
    results = run_genetic_algorithm(pop_size=50, generations=50)

    # Print results
    print("\nGenetic Algorithm Results:")
    print(f"Runtime: {results['runtime']:.2f} seconds")
    print(f"Best Fitness: {results['best_fitness']}")
    print(f"Best Strategy Genome: {[f'{g:.2f}' for g in results['best_strategy'].genome]}")

    # Print interpretation
    print("\nBest Strategy Interpretation:")
    for key, value in results['analysis']['interpretation'].items():
        print(f"  {key}: {value}")

    # Print performance against fixed strategies
    print("\nPerformance Against Fixed Strategies:")
    for opponent, scores in results['analysis']['performance'].items():
        print(f"  {opponent}: Strategy Score = {scores['strategy_score']}, Opponent Score = {scores['opponent_score']}")

    # Plot fitness history
    plot_fitness_history(results['best_fitness_history'], results['avg_fitness_history'])


if __name__ == "__main__":
    main()