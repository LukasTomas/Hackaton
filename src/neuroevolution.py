import time
import torch
import random
import numpy as np
import math
from deap import base, creator, tools, algorithms

import nn
from hyperparams import POPULATION_SIZE, GENERATIONS, CROSSOVER_PROB, MUTATION_PROB, SELECTION_SIZE, NN_INPUT_SIZE
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
        flat_params = [p.item() for tensor in params for p in tensor.flatten()]  # Flatten tensors
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
            mutation = random.uniform(-mutation_strength, mutation_strength)
            individual[i] += mutation  # Update the individual's parameter


rnd_tensor = torch.rand(1, NN_INPUT_SIZE)
evaluator = Evaluator(games_percent=0.01)
def evaluate(individual):
    neural_network = create_nn(individual)
    bankroll = evaluator.evaluate(neural_network)
    # prediction = neural_network(rnd_tensor)[0][0].item()

    # output = 1/(abs(prediction-0.69))
    return bankroll,


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

    # Roulette - assigns a higher probability of selection to individuals with higher fitness function
    #            values, but does not guarantee selection of the best individual
    # toolbox.register("select", tools.selRoulette)    
    toolbox.register("select", tools.selBest)

    toolbox.register("mutate", mutate)
    toolbox.register("mate", uniform_crossover)

    for gen_i in range(GENERATIONS):
        print(f"Generation {gen_i}", len(population))

        # evaluate each individual in the population
        for indiv in population:
            indiv.fitness.values = toolbox.evaluate(indiv)


        print("Population fitness:")
        for indiv in population:
            print(indiv.fitness.values[0], end=" ")

        # select the --hyperparameter-- best individuals
        best_indivs = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)[:SELECTION_SIZE]

        # Clone the best individuals
        best_indivs = list(map(toolbox.clone, best_indivs))

        print("\nBest individuals fitness:")
        for indiv in best_indivs:
            print(indiv.fitness.values[0], end=" ")
            del indiv.fitness.values
        print()
        neural_network = create_nn(best_indivs[0])
        prediction = neural_network(rnd_tensor)[0][0].item()
        print("Best model is outputting: ", prediction)
        print()

        # crossover the best individuals
        offspring = []
        while len(offspring) + len(best_indivs) < POPULATION_SIZE:
            # Select pairs of parents from the best individuals for crossover
            parent1, parent2 = random.sample(best_indivs, 2)

            # Apply crossover and mutation
            child1, child2 = toolbox.mate(parent1, parent2)
            toolbox.mutate(child1, mutation_strength=0.1)
            # toolbox.mutate(child2, mutation_strength=0.1)

            # Invalidate fitness for children to ensure they're re-evaluated
            del child1.fitness.values
            del child2.fitness.values

            # Add children to offspring, checking not to exceed population size
            offspring.extend([child1, child2])

        # Trim any extra offspring if we exceeded the population size
        offspring = offspring[:POPULATION_SIZE - len(best_indivs)]


        # Combine best individuals and offspring to form the new population
        population = best_indivs + offspring

        assert len(population) == POPULATION_SIZE


    for indiv in population:
        indiv.fitness.values = toolbox.evaluate(indiv)

    # Output the best individual from the final population
    best_individual = tools.selBest(population, 1)[0]
    print(f"Best individual fitness: {best_individual.fitness.values}")
    best_nn = create_nn(best_individual)
    # save nn to file
    torch.save(best_nn.state_dict(), 'best_nn.pth')

