import numpy as np
import random
import copy

class Genome():
    def __init__(self, weights, biases):
        # A genome is one candidate neural network: all weights and biases.
        self.fitness = 0
        self.weights = weights
        self.biases = biases
        
    def mutate(self):
        # Randomly nudge genes so the next generation explores new controllers.
        mutation_rate = 0.95
        for i in range(0, len(self.weights)):
            if np.random.random() < mutation_rate:
                self.weights[i] += np.random.uniform(-1, 1) * 0.075
        
        for i in range(0, len(self.biases)):
            if np.random.random() < mutation_rate:
                self.biases[i] += np.random.uniform(-1, 1) * 0.20    

    def __lt__(self, other_genome):
        return self.fitness < other_genome.fitness

class Genetic_algorithm():
    def __init__(self, population_size, number_of_weights, number_of_biases):
        # Start with a population of random neural-network genomes.
        self.population_size = population_size
        self.population = [None] * population_size
        self.number_of_weights = number_of_weights
        self.number_of_biases = number_of_biases

        for i in range(population_size):
            initial_weights = np.random.uniform(-1, 1, number_of_weights).tolist()
            initial_biases = np.random.uniform(-1, 1, number_of_biases).tolist()
            self.population[i] = Genome(initial_weights, initial_biases)
    
    def get_genome_by_tournament(self):
        # Pick a small random group and return the best genome from that group.
        tournament_size = 4
        combatants = np.random.choice(range(0, len(self.population)), tournament_size, replace = False)
        combatants.sort()
        fittest_genome = self.population[combatants[0]]      

        return fittest_genome

    def crossover(self, parent_0, parent_1):
        # Create one child by combining a prefix from one parent and a suffix from the other.
        crossover_rate = 0.95
        parents = np.random.choice([parent_0, parent_1], 2, replace = False)

        if np.random.random() < crossover_rate:
            random_weight_index = random.randint(0, self.number_of_weights)
            random_bias_index = random.randint(0, self.number_of_biases)
            child = Genome([None] * self.number_of_weights, [None] * self.number_of_biases)

            child.weights[0:random_weight_index] = parents[0].weights[0:random_weight_index]
            child.weights[random_weight_index::] = parents[1].weights[random_weight_index::]
            child.biases[0:random_bias_index] = parents[0].biases[0:random_bias_index]
            child.biases[random_bias_index::] = parents[1].biases[random_bias_index::]
            
            return child
        else:
           return copy.deepcopy(parents[0])

    def update(self, agents):
        # Copy survival scores from agents back into their matching genomes.
        for i, agent in enumerate(agents):
            self.population[i].fitness = agent.fitness
    
    def upgrade(self):
        # Sort by fitness, keep the best genomes, then fill the rest with children.
        self.population.sort(reverse = True)
        
        new_population = [None] * self.population_size
        new_population[0] = self.population[0]
        new_population[1] = self.population[1]
        new_population[2] = self.population[2]
        new_population[3] = self.crossover(self.population[0], self.population[1])
        new_population[4] = self.crossover(self.population[0], self.population[2])
        new_population[5] = self.crossover(self.population[1], self.population[2]) 
        
        for i in range(6, self.population_size):
            parent_0 = self.get_genome_by_tournament()
            parent_1 = self.get_genome_by_tournament()
            new_population[i] = self.crossover(parent_0, parent_1)					

        for i in range(1, self.population_size):
            new_population[i].mutate() 
                
        # Store all new genomes
        self.population = new_population
