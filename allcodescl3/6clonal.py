import numpy as np
import random

def rastrigin(x):
    A = 10
    return A * len(x) + sum(x_i**2 - A * np.cos(2 * np.pi * x_i) for x_i in x)

def clonal_selection_algorithm(pop_size, generations, mutation_rate, elite_size):
    population = np.random.uniform(-5.12, 5.12, (pop_size, 2))
    for gen in range(generations):

        fitness = np.array([rastrigin(ind) for ind in population])
        elite_indices = np.argsort(fitness)[:elite_size] 
        elite_individuals = population[elite_indices]

       
        clones = elite_individuals.copy()

        for i in range(len(clones)):
            if random.random() < mutation_rate:
                clones[i] += np.random.uniform(-0.1, 0.1, 2)  
                clones[i] = np.clip(clones[i], -5.12, 5.12)  

        worst_indices = np.argsort(fitness)[-elite_size:]  
        population[worst_indices] = clones

        best_solution = population[np.argmin(fitness)]  
        print(f"Generation {gen+1}, Best Solution: {best_solution}, Fitness: {rastrigin(best_solution)}")

    return population

population_size = 120  
generations = 100  
mutation_rate = 0.1 
elite_size = 4  

# clonal_selection_algorithm(population_size, generations, mutation_rate, elite_size)

final_population = clonal_selection_algorithm(population_size, generations, mutation_rate, elite_size)
final_fitness = np.array([rastrigin(ind) for ind in final_population])
best_index = np.argmin(final_fitness)
print(f"\nFinal Best Solution: {final_population[best_index]}")
print(f"Final Best Fitness: {final_fitness[best_index]}")
