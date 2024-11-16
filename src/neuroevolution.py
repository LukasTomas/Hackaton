import time
import math
import glob
import torch
import random
import threading
import numpy as np
from collections import deque
from deap import base, creator, tools, algorithms

from evaluation import Evaluator
import nn
from hyperparams import POPULATION_SIZE, GENERATIONS, CROSSOVER_PROB, MUTATION_PROB, SELECTION_SIZE, NN_INPUT_SIZE, \
    MUTATION_STRENGTH, ELITE_COUNT

MODEL_NAME = "best_nn_"


def create_nn(individual):
     # Initialize a neural network with the correct architecture
    neural_network = nn.LinearNN()
    
    # Prepare state_dict in the correct format for loading
    state_dict = neural_network.state_dict()
    param_shapes = [param.shape for param in state_dict.values()]
    
    # Reshape individual parameters into tensors matching state_dict structure
    reshaped_params = []
    start = 0
    for shape in param_shapes:
        size = torch.prod(torch.tensor(shape)).item()
        reshaped_tensor = torch.tensor(individual[start:start + size]).view(shape)
        reshaped_params.append(reshaped_tensor)
        start += size

    # Load parameters into the model
    state_dict = {k: v for k, v in zip(state_dict.keys(), reshaped_params)}
    neural_network.load_state_dict(state_dict)
    return neural_network

def init_population():
    population = []

    saved_models = glob.glob("*.pth")
    print(f'Loading {len(saved_models)} saved models')

    for model_path in saved_models:
        neural_network = nn.LinearNN()
        neural_network.load_state_dict(torch.load(model_path))
        
        params = list(neural_network.state_dict().values())
        flat_params = torch.cat([p.flatten() for p in neural_network.parameters()]).tolist()
        
        individual = creator.Individual(flat_params)
        population.append(individual)

    init_rest_number = POPULATION_SIZE - len(saved_models)
    for _ in range(init_rest_number):
        neural_network = nn.LinearNN()  # Initialize the neural network
        params = list(neural_network.state_dict().values())
        # Initialize neural network and get flattened parameters more cleanly
        flat_params = torch.cat([p.flatten() for p in neural_network.parameters()]).tolist()

        individual = creator.Individual(flat_params)  # Create individual with flattened params
        population.append(individual)
    return population

def mutate(individual, mutation_rate=0.1, mutation_strength=0.1):
    """
    Mutates an individual by modifying its parameters with a specified mutation rate and strength.

    Parameters:
    - individual: The individual to mutate.
    - mutation_rate: The probability of mutating each parameter.
    - mutation_strength: The maximum amount by which a parameter can be mutated.
    """
    # Iterate over the individual's parameters
    for i in range(len(individual)):
        if random.random() < mutation_rate:  # Decide whether to mutate this parameter
            # Apply a small random change to the parameter
            mutation = random.gauss(0, mutation_strength)
            individual[i] += mutation  # Update the individual's parameter

def evaluate(individual):
    neural_network = create_nn(individual)
    evaluator = Evaluator(games_percent=0.01, games_percentage_start=0.8)
    bankroll = evaluator.evaluate(neural_network)
    return bankroll,

def blx_alpha_crossover(parent1, parent2, alpha=0.5):
    child1, child2 = [], []
    for p1, p2 in zip(parent1, parent2):
        min_val, max_val = min(p1, p2), max(p1, p2)
        interval = max_val - min_val
        lower, upper = min_val - interval * alpha, max_val + interval * alpha
        child1.append(random.uniform(lower, upper))
        child2.append(random.uniform(lower, upper))
    return creator.Individual(child1), creator.Individual(child2)

def uniform_crossover(parent1, parent2, prob=0.5):
    child1, child2 = parent1[:], parent2[:]
    for k in range(len(parent1)):
        if random.random() < prob:
            child1[k], child2[k] = child2[k], child1[k]  # Swap parameter values
    return creator.Individual(child1), creator.Individual(child2)

def thread_evaluate(deque_population, deque_lock, toolbox, thrad_id):
    while True:
        with deque_lock:
            if len(deque_population) == 0:
                return

            inidividual = deque_population.popleft()
    
        # print(f"Thread {thrad_id} choosing individual {inidividual[0]}")
        inidividual.fitness.values = toolbox.evaluate(inidividual)
        # print(f"Thread {thrad_id} finished evaluating individual {inidividual.fitness.values[0]}")

if __name__ == "__main__":
    random.seed(time.time())
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    population = init_population()

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)

    toolbox.register("select", tools.selTournament, tournsize=3)

    # toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)
    toolbox.register("mutate", mutate)
    # toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mate", uniform_crossover)

    threads_number = 8

    for gen_i in range(GENERATIONS):
        print(f"Generation {gen_i}", len(population))


        # Step 1: Evaluate each individual in the population
        threads = []

        # deque_population = deque(enumerate(population))
        deque_population = deque(population)

        deque_lock = threading.Lock()
        for thread_id in range(threads_number):
            thread = threading.Thread(target=thread_evaluate, args=(deque_population, deque_lock, toolbox, thread_id))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()

        fitness_vals = []
        for indiv in population:
            fitness_vals.append(indiv.fitness.values[0])

        # Save the best individuals from the current population
        number_best = 5
        best_individuals = tools.selBest(population, number_best)
        for i in range(len(best_individuals)):
            if best_individuals[i].fitness.values[0] < 1000:
                continue
            best_nn = create_nn(best_individuals[i])
            torch.save(best_nn.state_dict(), f'{MODEL_NAME}{i}_{int(best_individuals[i].fitness.values[0])}.pth')

        # Step 2: Tournament Selection
        selected_indivs = toolbox.select(population, SELECTION_SIZE)
        min_fitness = min(ind.fitness.values[0] for ind in selected_indivs)
        weights = [(ind.fitness.values[0] - min_fitness + 1e-6) for ind in selected_indivs]

        # Step 3: Elitism - Preserve a small number of the best individuals
        elites = tools.selBest(population, ELITE_COUNT)

        print(f"Generation {gen_i} - Max: {max(fitness_vals)}, Min: {min(fitness_vals)}, Mean: {np.mean(fitness_vals)}")
        # Step 4: Weighted Selection for Parents Based on Profit (Fitness)
        offspring = []
        # for elite in elites:
        #     offspring.extend(list(map(toolbox.clone, elites)))

        # Calculate alpha based on current

        while len(offspring) + len(elites) < POPULATION_SIZE:
            # Select pairs of parents from the best individuals for crossover
            parent1, parent2 = random.choices(selected_indivs, weights=weights, k=2)

            # Apply crossover and mutation
            child1, child2 = toolbox.mate(parent1, parent2)

            # Add children to offspring, checking not to exceed population size
            offspring.extend([child1, child2])

        # Trim any extra offspring if we exceeded the population size
        offspring = offspring[:POPULATION_SIZE - len(elites)]

        # Step 5: Adaptive Mutation for Offspring
        mutation_strength = MUTATION_STRENGTH * (1 - gen_i / GENERATIONS)
        for child in offspring:
            toolbox.mutate(child, MUTATION_PROB, mutation_strength)
            # toolbox.mutate(child, sigma=mutation_strength, indpb=MUTATION_PROB)
            del child.fitness.values

        # Combine best individuals and offspring to form the new population
        population = elites + offspring

        assert len(population) == POPULATION_SIZE


    for indiv in population:
        indiv.fitness.values = toolbox.evaluate(indiv)

    # Output the best individual from the final population
    best_individual = tools.selBest(population, 1)[0]
    print(f"Best individual fitness: {best_individual.fitness.values}")
    best_nn = create_nn(best_individual)
    # save nn to file
    torch.save(best_nn.state_dict(), 'best_nn.pth')

