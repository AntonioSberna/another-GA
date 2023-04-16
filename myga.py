





import numpy as np
import random
from functools import partial



class MyGA:


    def __init__(self, individual, population_size, gene_mask, **kwargs):
        self.individual = individual
        self.population_size = population_size
        self.gene_mask = gene_mask
        self.population = []
        self.offspring_size = kwargs.get('offspring_size', population_size)
        self.tourn_size = kwargs.get('tourn_size', 4)

        self.generation = 0
        self.stall = 0
        

## -----------------------------------------------------------------------------------------------------------------
## FITNESS FUNCTION

    def fitness_function(self, individual):
        # Here should be all the part of structural analysis and fitness evaluation

        # for example here I've set the Himmelblau's function
        # https://en.wikipedia.org/wiki/Himmelblau%27s_function

        x = individual.genotype[0]
        y = individual.genotype[1]
        return (x**2+y-11)**2 + (x+y**2-7)**2



## -----------------------------------------------------------------------------------------------------------------
## POPULATION CREATOR

    # The sampling of the research space for creating the first population
    def initial_population(self, **kwargs):
        # dimension of the initial population with respect pop_size 
        k = kwargs.get('k', 1)

        for _ in range(self.population_size*k):
            genome = self.random_genome(self.gene_mask)
            # self.population.append(self.individual(ind, [genome, np.nan]))
            self.population.append(self.individual(genome, np.nan))
    
    # Create a new population of "pop_dim" individuals starting from "population"
    def new_population(self, population, **kwargs):
        pop_dim = kwargs.get('pop_dim', self.population_size)

        new_population = []
        for _ in range(pop_dim):
            # Select the parents
            parents = self.parent_selection(population, n_par = 2, tourn_size = self.tourn_size)
            # Crossover of the parents genome
            offspring = self.crossover(parents, k = 1)
            # Mutation
            new_population.append(self.mutation(offspring))

        return new_population




## -----------------------------------------------------------------------------------------------------------------
## GENETIC OPERATORS (and other subroutines used)


    def crossover(self, parents, **kwargs):
        k = kwargs.get('k', 1)

        offspring = self.kpointcrossover(self, parents, k = k)
        return offspring
    


    def mutation(self, individual):

        mutated_individual = self.onepointmutation(individual, self.gene_mask)
        return mutated_individual
    

    # Select the parents for the crossover
    def parent_selection(self, population, **kwargs):
        n_par = kwargs.get('n_par', 2)
        tourn_size = kwargs.get('tourn_size')

        return self.tournament_selection(population, n_par, tourn_size)


    # At the end of each generation, take the best individuals
    def survival_selection(self):
        self.population = self.sort_truncate(self, self.population)


    # Check if the stopping criteria are reached
    def stopping_criteria(self, stop_crit):
        viol = []
        for name, cond in stop_crit.items():
            viol.append(eval(f"{name} {cond['operator']} {cond['value']}"))
        return [ind for ind, cond in enumerate(viol) if cond]








## -----------------------------------------------------------------------------------------------------------------
## STATIC METHODS



    # GENERATION OF RANDOM INDIVIDUAL
    @staticmethod
    def random_genome(gene_mask):
        return [random.choice(arr) for arr in gene_mask]


    # SURVIVAL SELECTION
    @staticmethod
    def sort_truncate(self, population):
        # This is deterministic for now, you can use some function that has some slight randomness
        # reverse = False -> minimization problem
        return sorted(population, key=lambda i: i.fitness, reverse=False)[:self.population_size]
    

    # PARENT SELECTION
    @staticmethod
    def tournament_selection(population, n_par, tourn_size):
        # Selection of the partecipant to the tournament
        tournament = random.choices(population, k=tourn_size)

        # Take the best "n_par" individuals partecipating at the tournament (accordin to their fitness value)
        return list(sorted(tournament, key=lambda i: i.fitness, reverse=False)[:n_par])
        
    


    @staticmethod
    def kpointcrossover(self, parents, **kwargs):
        k = kwargs.get('k', 1)

        # Take the point where to cut the genome 
        cut_points = sorted(random.sample(range(1,len(parents[0].genotype)), k))
        # Order of the parents (otherwise the first part of the genome will always come from the best performing parent)
        rand_ord = random.sample(range(2), 2)

        # Create offspring from the first parent (according to rand_ord)
        offspring = parents[rand_ord[0]]

        # Swap piece of genome according to cut_points 
        for i in range(len(cut_points)):
            start = cut_points[i]
            end = cut_points[i+1] if i+1 < len(cut_points) else len(parents[0].genotype)
            offspring.genotype[start:end] = parents[rand_ord[1]].genotype[start:end]

        # Remove fitness value
        return offspring._replace(fitness=np.nan)
    

    @staticmethod
    def onepointmutation(individual, gene_mask):
        # One random point inside the genome
        rand_ind = random.sample(range(len(individual.genotype)), 1)[0]
        # Change the value to that random sample
        individual.genotype[rand_ind] = random.choice(gene_mask[rand_ind])
        return individual

    @staticmethod
    def run_fitness_evaluation(fitness_function, population, individual):
        #this makes it easy to parallelize the analyses (just use a parallelized mapping function (be careful how you handle opensees!))
        to_run = partial(fitness_function)
        fitness = list(map(to_run, population))
        return [individual(ind.genotype, fit) for ind, fit in zip(population, fitness)]
        # or:
        # return [individual(ind.genotype, fit) for ind, fit in zip(myga.population, list(map(partial(fitness_function), myga.population)))]
    