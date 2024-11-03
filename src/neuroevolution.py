import time
import torch
import random
import numpy as np
import math
from deap import base, creator, tools, algorithms

import nn
from hyperparams import POPULATION_SIZE, GENERATIONS, CROSSOVER_PROB, MUTATION_PROB, SELECTION_SIZE, NN_INPUT_SIZE, \
    MUTATION_STRENGTH, ELITE_COUNT
from evaluation import Evaluator


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

    for _ in range(POPULATION_SIZE):
        neural_network = nn.LinearNN()  # Initialize the neural network
        params = list(neural_network.state_dict().values())
        # Initialize neural network and get flattened parameters more cleanly
        flat_params = torch.cat([p.flatten() for p in neural_network.parameters()]).tolist()

        individual = creator.Individual(flat_params)  # Create individual with flattened params
        population.append(individual)
    return population

random.seed(time.time())


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


rnd_tensor = torch.rand(1, NN_INPUT_SIZE)
evaluator = Evaluator(games_percent=0.01)
def evaluate(individual):
    neural_network = create_nn(individual)
    bankroll = evaluator.evaluate(neural_network)
    #
    # prediction = neural_network(rnd_tensor)[0][0].item()
    # output = 1 / (abs(prediction - 0.69) + 1e-6)

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


if __name__ == "__main__":
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
    for gen_i in range(GENERATIONS):
        print(f"Generation {gen_i}", len(population))


        # Step 1: Evaluate each individual in the population
        fitness_vals = []
        for indiv in population:
            indiv.fitness.values = toolbox.evaluate(indiv)
            fitness_vals.append(indiv.fitness.values[0])

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

