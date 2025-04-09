import argparse
import csv
import math
import os
import tsp_ga  # Import the GA module
import matplotlib.pyplot as plt
from collections import defaultdict


def run_experiments(datasets, pop_sizes, crossover_rates, mutation_rates, generations, num_runs):
    """
    Runs experiments on multiple datasets using various parameter combinations.
    Returns a list of result dictionaries.
    """
    results = []
    for dataset in datasets:
        coords = tsp_ga.read_tsplib(dataset)
        if not coords:
            print(f"Error reading {dataset}")
            continue
        dist_matrix = tsp_ga.create_distance_matrix(coords)
        num_cities = len(coords)
        print(f"\nRunning experiments on {os.path.basename(dataset)} with {num_cities} cities:")

        for pop in pop_sizes:
            for cr in crossover_rates:
                for mr in mutation_rates:
                    run_lengths = []
                    run_times = []
                    for run in range(num_runs):
                        _, L, _, t = tsp_ga.run_ga(dist_matrix, pop_size=pop, generations=generations,
                                                   crossover_rate=cr, mutation_rate=mr)
                        run_lengths.append(L)
                        run_times.append(t)
                        print(f"Dataset: {os.path.basename(dataset)}, pop: {pop}, cr: {cr}, mr: {mr}, "
                              f"Run {run + 1}: Length = {L:.2f}, Time = {t:.2f}s")
                    avg_length = sum(run_lengths) / num_runs
                    avg_time = sum(run_times) / num_runs
                    std_length = math.sqrt(sum((x - avg_length) ** 2 for x in run_lengths) / num_runs)
                    results.append({
                        "dataset": os.path.basename(dataset),
                        "cities": num_cities,
                        "pop_size": pop,
                        "crossover_rate": cr,
                        "mutation_rate": mr,
                        "avg_length": avg_length,
                        "std_length": std_length,
                        "avg_time": avg_time
                    })
    return results


def save_results_to_csv_by_dataset(results):
    """
    Saves separate CSV files for each dataset.
    """
    grouped = defaultdict(list)
    for r in results:
        grouped[r["dataset"]].append(r)
    for dataset, res_list in grouped.items():
        filename = f"{os.path.splitext(dataset)[0]}_results.csv"
        fieldnames = ["dataset", "cities", "pop_size", "crossover_rate", "mutation_rate",
                      "avg_length", "std_length", "avg_time"]
        with open(filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(res_list)
        print(f"Results for {dataset} saved to {filename}")


def plot_results(results):
    """
    Creates a line graph for each dataset showing the best (lowest) average tour length vs. population size.
    For each dataset, for every population size, the best average length (across all crossover and mutation settings) is plotted.
    """
    # Group results by dataset and then by population size
    dataset_data = defaultdict(dict)
    for r in results:
        dataset = r["dataset"]
        pop = r["pop_size"]
        # Choose the best (lowest) avg_length for this population size if multiple entries exist
        if pop in dataset_data[dataset]:
            dataset_data[dataset][pop] = min(dataset_data[dataset][pop], r["avg_length"])
        else:
            dataset_data[dataset][pop] = r["avg_length"]

    plt.figure(figsize=(10, 6))
    for dataset, pop_dict in dataset_data.items():
        pop_sizes = sorted(pop_dict.keys())
        avg_lengths = [pop_dict[pop] for pop in pop_sizes]
        plt.plot(pop_sizes, avg_lengths, marker='o', label=dataset)
    plt.xlabel("Population Size")
    plt.ylabel("Best Average Tour Length")
    plt.title("Effect of Population Size on TSP Performance")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="TSP GA Experiment Runner")
    parser.add_argument("--datasets", nargs="+", default=["berlin52.tsp", "kroA100.tsp", "pr1002.tsp"],
                        help="List of TSPLIB dataset files")
    parser.add_argument("--pop_sizes", nargs="+", type=int, default=[50, 100, 200],
                        help="Population sizes to test")
    parser.add_argument("--crossover_rates", nargs="+", type=float, default=[0.6, 0.8, 1.0],
                        help="Crossover rates to test")
    parser.add_argument("--mutation_rates", nargs="+", type=float, default=[0.1, 0.2, 0.3],
                        help="Mutation rates to test")
    parser.add_argument("--generations", type=int, default=500,
                        help="Number of generations")
    parser.add_argument("--num_runs", type=int, default=10,
                        help="Number of independent runs per parameter configuration")
    args = parser.parse_args()

    results = run_experiments(args.datasets, args.pop_sizes, args.crossover_rates,
                              args.mutation_rates, args.generations, args.num_runs)
    save_results_to_csv_by_dataset(results)
    plot_results(results)


if __name__ == '__main__':
    main()
