import random
from deap import base, creator, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    return [random.randint(0, 1) for _ in range(10)]  

def evaluate(individual):
    return sum(individual),  

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population = toolbox.population(n=30)

algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=10,verbose=True)

best_individual = tools.selBest(population, 1)[0]
print("Best Individual:", best_individual)