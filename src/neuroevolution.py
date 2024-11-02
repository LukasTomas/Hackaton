import time
import torch
import random
import numpy as np
import math
from deap import base, creator, tools, algorithms

import nn
from hyperparams import POPULATION_SIZE, GENERATIONS, CROSSOVER_PROB, MUTATION_PROB, SELECTION_SIZE, NN_INPUT_SIZE


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
    # for _ in range(POPULATION_SIZE):
    #     neural_network = nn.LinearNN()
    #     params = neural_network.state_dict()
    #     individual = creator.Individual([params.values()])
        # population.append(individual)    

    for _ in range(POPULATION_SIZE):
        neural_network = nn.LinearNN()  # Initialize the neural network
        params = list(neural_network.state_dict().values())
        flat_params = [p.item() for tensor in params for p in tensor.flatten()]  # Flatten tensors
        individual = creator.Individual(flat_params)  # Create individual with flattened params
        population.append(individual)
    return population

random.seed(time.time())


def mutate(individual):
    for i in range(len(individual)):
        # TODO dynamically determine the mutation range
        individual[i] += random.gauss(0, 0.05) 
    return individual,


rnd_tensor = torch.rand(1, NN_INPUT_SIZE)
def evaluate(individual):
    neural_network = create_nn(individual)
    
    output = neural_network(rnd_tensor)
    return output,  


def crossover(parent1, parent2):
    return tools.cxUniform(parent1, parent2, indpb=0.5)


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
    # toolbox.register("mate", crossover)

    parents_number_childs = POPULATION_SIZE / (SELECTION_SIZE / 2)
    parents_childs = []
    total_childs = 0
    for i in range(int(SELECTION_SIZE/2)):
        childs_number = math.ceil(parents_number_childs)
        if total_childs + childs_number > POPULATION_SIZE:
            childs_number = POPULATION_SIZE - total_childs

        parents_childs.append(childs_number)
        total_childs += childs_number
    print(parents_childs)

    for gen_i in range(GENERATIONS):
        print(f"Generation {gen_i}", len(population))

        # evaluate each individual in the population
        for indiv in population:
            indiv.fitness.values = toolbox.evaluate(indiv)


        print("Population fitness:")
        for indiv in population:
            print(indiv.fitness.values[0][0][0].item(), end=" ")

        # select the --hyperparameter-- best individuals
        best_indivs = toolbox.select(population, SELECTION_SIZE)
        best_indivs = list(map(toolbox.clone, best_indivs))

        print("\nBest individuals fitness:")
        for indiv in best_indivs:
            print(indiv.fitness.values[0][0][0].item(), end=" ")   
        print()


        # crossover the best individuals
        # population = []
        # parent_index = 0
        # for child1, child2 in zip(best_indivs[::2], best_indivs[1::2]):
        #     childs_number = int(parents_childs[parent_index] / 2)
        #     for _ in range(childs_number):
        #         child1 = toolbox.clone(child1)
        #         child2 = toolbox.clone(child2)
        #         del child1.fitness.values
        #         del child2.fitness.values
                
        #         if np.random.rand() < CROSSOVER_PROB:
        #             # toolbox.mate(child1, child2)
        #             tools.cxUniform(child1, child2, indpb=0.5)

        #         population.append(child1)
        #         population.append(child2)

        #     parent_index += 1

        # mutate the best individuals
        for mutant in population:
            if np.random.rand() < MUTATION_PROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        assert len(population) == POPULATION_SIZE


        for indiv in population:
            indiv.fitness.values = toolbox.evaluate(indiv)

        print("After crossover and mutation:")
        for indiv in population:
            print(indiv.fitness.values[0][0][0].item(), end=" ")
        print()

        best_individual = tools.selBest(population, 1)[0]
        print(f"Best individual fitness: {best_individual.fitness.values[0][0][0].item()}")
        print()


    for indiv in population:
            indiv.fitness.values = toolbox.evaluate(indiv)

    # Output the best individual from the final population
    best_individual = tools.selBest(population, 1)[0]
    print(f"Best individual fitness: {best_individual.fitness.values}")
    best_nn = create_nn(best_individual)
    # save nn to file
    torch.save(best_nn.state_dict(), 'best_nn.pth')


# import time
# import torch
# import random
# import numpy as np
# import math
# from deap import base, creator, tools, algorithms

# import nn  # Import your neural network architecture
# from hyperparams import POPULATION_SIZE, GENERATIONS, CROSSOVER_PROB, MUTATION_PROB, SELECTION_SIZE, NN_INPUT_SIZE

# # Initialize the random seed
# random.seed(time.time())

# # Create a DEAP Individual and Fitness classes
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)

# def init_population():
#     """Initialize the population with individuals containing neural network parameters."""
#     population = []
#     for _ in range(POPULATION_SIZE):
#         neural_network = nn.LinearNN()  # Initialize the neural network
#         params = list(neural_network.state_dict().values())
#         flat_params = [p.item() for tensor in params for p in tensor.flatten()]  # Flatten tensors
#         individual = creator.Individual(flat_params)  # Create individual with flattened params
#         population.append(individual)
#     return population

# def mutate(individual):
#     """Mutate the individual's parameters by adding Gaussian noise."""
#     for i in range(len(individual)):
#         # Mutate with a standard deviation of 0.05
#         individual[i] += random.gauss(0, 0.05)
#     return individual,

# rnd_tensor = torch.rand(1, NN_INPUT_SIZE)

# def evaluate(individual):
#     """Evaluate the individual's fitness by running the neural network."""
#     neural_network = nn.LinearNN()
#     state_dict = neural_network.state_dict()
#     param_shapes = [param.shape for param in state_dict.values()]

#     reshaped_params = []
#     start = 0
#     for shape in param_shapes:
#         size = torch.prod(torch.tensor(shape)).item()
#         reshaped_tensor = torch.tensor(individual[start:start + size]).view(shape)
#         reshaped_params.append(reshaped_tensor)
#         start += size

#     state_dict = {k: v for k, v in zip(state_dict.keys(), reshaped_params)}
#     neural_network.load_state_dict(state_dict)
    
#     output = neural_network(rnd_tensor)
#     return output,

# def main():
#     # Initialize population
#     population = init_population()

#     # Setup DEAP toolbox
#     toolbox = base.Toolbox()
#     toolbox.register("evaluate", evaluate)
#     toolbox.register("select", tools.selBest)
#     toolbox.register("mate", tools.cxUniform, indpb=CROSSOVER_PROB)
#     toolbox.register("mutate", mutate)

#     # Run the evolutionary algorithm
#     for gen_i in range(GENERATIONS):
#         print(f"Generation {gen_i}, Population size: {len(population)}")

#         # Evaluate individuals
#         fitnesses = list(map(toolbox.evaluate, population))
#         for ind, fit in zip(population, fitnesses):
#             ind.fitness.values = fit

#         # Select the next generation individuals
#         offspring = toolbox.select(population, SELECTION_SIZE)
#         offspring = list(map(toolbox.clone, offspring))

#         # Apply crossover and mutation
#         for child1, child2 in zip(offspring[::2], offspring[1::2]):
#             if np.random.rand() < CROSSOVER_PROB:
#                 toolbox.mate(child1, child2)
#                 del child1.fitness.values
#                 del child2.fitness.values

#         for mutant in offspring:
#             if np.random.rand() < MUTATION_PROB:
#                 toolbox.mutate(mutant)
#                 del mutant.fitness.values

#         # Evaluate the new individuals
#         fitnesses = list(map(toolbox.evaluate, offspring))
#         for ind, fit in zip(offspring, fitnesses):
#             ind.fitness.values = fit

#         # Replace the old population by the offspring
#         population[:] = offspring

#         # Print fitness statistics
#         fits = [ind.fitness.values[0].item() for ind in population]
#         print(f"Best fitness: {max(fits)}")
#         print(f"Average fitness: {np.mean(fits)}")
#         print(f"Worst fitness: {min(fits)}")
#         print()

#     # Final evaluation of the best individual
#     best_individual = tools.selBest(population, 1)[0]
#     print(f"Best individual fitness: {best_individual.fitness.values}")

# if __name__ == "__main__":
#     main()
