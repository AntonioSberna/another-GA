

############################################################################
#####                                                                  #####
#####                Steady-state Genetic Algorithm                    #####
#####                                                                  #####
#####antonio DOT sberna AT polito DOT it####################################


# This is a test of a Genetic algorithm for minimizing a function
# the first population is created sampling randomly the research space
# the other populations are created according to the steady state approach
# with a tournament parent selection, k-points crossover, and single locus 
# random drawing mutation. 



import numpy as np
from collections import namedtuple
from myga import MyGA

### Initial inputs

# Number of individuals in a population
population_size = 200
# Number of offspring
offspring_size = 200
# Dimension of the tournament
tourn_size = 5
# Mask of the values that the parameters can assume 
gene_mask = (np.linspace(start=-6, stop=6, num = 200), 
             np.linspace(start=-6, stop=6, num = 200))
# the number of decision variables is implicitly set in the mask (# rows)


# Creation structure of each individual (with a genotype and a fitness)
individual = namedtuple('individual', ['genotype', 'fitness'])
individual.__new__.__defaults__ = (np.nan,)


# Instance of MyGa
myga = MyGA(individual, population_size, gene_mask, offspring_size = offspring_size, tourn_size = tourn_size)

# Stopping criteria 
# in this example, if the generation is greater than 10, stall than 5 and the best individual of this generation smaller than 0.1
stop_crit = {"self.generation": {"value": 100, "operator": ">="},
             "self.stall": {"value": 5, "operator": ">"},
             "self.population[0].fitness": {"value": 0.1, "operator": "<="}}



### Initial population 
# Creation of initial population (just the genome)
myga.initial_population(k=2)

# Evaluating fitness for the initial population (separated since, in our case, this is the time consuming part)
myga.population = myga.run_fitness_evaluation(myga.fitness_function, myga.population, individual)

# Survival selection for the initial population 
myga.survival_selection()


### Other populations
while not myga.stopping_criteria(stop_crit):
    # Increment the generation counter
    myga.generation +=1

    # Create a new population
    new_population = myga.new_population(myga.population)

    # Evaluate fitness of the indiv. of this new population
    new_population = myga.run_fitness_evaluation(myga.fitness_function, new_population, individual)

    # Append the new population to the previous one 
    myga.population.extend(new_population)

    # Survival selection
    myga.survival_selection()

    # Increment stall (if the fitness of the current best individual is the same of the one on the previous generation)
    if new_population[0].fitness == myga.population[0].fitness:
        myga.stall += 1
    else:
        myga.stall = 0

