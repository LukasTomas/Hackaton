import torch
from deap import base, creator, tools, algorithms

import nn
from hyperparams import POPULATION_SIZE, GENERATIONS, MUTATION_PROB, NN_INPUT_SIZE


def init_population():
    population = []
    for _ in range(POPULATION_SIZE):
        neural_network = nn.LinearNN()
        params = neural_network.state_dict()
        individual = creator.Individual([params.values()])
        population.append(individual)    
    return population


def evaluate(individual):


if __name__ == "__main__":
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    population = init_population()

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate, data=)
    toolbox.register("select", tools.selTournament, tournsize=3)    

    for gen_i in range(POPULATION_SIZE):
        print(f"Generation {gen_i}")
        best_indivs = toolbox.select(population, len(population))
        break