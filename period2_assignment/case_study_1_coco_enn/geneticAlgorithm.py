import copy
import numpy as np
import random

class Genome():
    def __init__(self, weights, biases):
        self.fitness = 0
        self.weights = weights
        self.biases = biases
        
    def mutate(self):
        MUTATION_RATE = 0.95
        LEARNING_RATE = 0.20
        for i in range(0, len(self.weights)):
            if np.random.random() < MUTATION_RATE:
                self.weights[i] += np.random.uniform(-1, 1) * LEARNING_RATE       
        
        for i in range(0, len(self.biases)):
            if np.random.random() < MUTATION_RATE:
                self.biases[i] += np.random.uniform(-1, 1) * LEARNING_RATE    
	
    def __lt__(self, other_genome):
        return self.fitness < other_genome.fitness

class Genetic_algorithm():
    def __init__(self, population_size, number_of_weights, number_of_biases):
        self.population_size = population_size
        self.population = np.full(population_size, None)
        self.number_of_weights = number_of_weights
        self.number_of_biases = number_of_biases

        for i in range(population_size):
            initial_weights = np.random.uniform(-1, 1, number_of_weights)
            initial_biases = np.random.uniform(-1, 1, number_of_biases)
            self.population[i] = Genome(initial_weights, initial_biases)            
   
    def get_genome_by_tournament(self):
        tournament_size = 4
        combatant_indices = np.random.choice(
            range(len(self.population)),
            tournament_size,
            replace=False
        )

        return max(
            (self.population[i] for i in combatant_indices),
            key=lambda genome: genome.fitness
        )

    def crossover(self, parent_0, parent_1):
        CROSSOVER_RATE = 0.95
        
        parents = [parent_0, parent_1]        
        np.random.shuffle(parents)

        if np.random.random() < CROSSOVER_RATE:
            random_weight_index = random.randint(0, self.number_of_weights)
            random_bias_index = random.randint(0, self.number_of_biases)
            child = Genome(np.full(self.number_of_weights, None), 
                           np.full(self.number_of_biases, None))

            child.weights[:random_weight_index] = parents[0].weights[:random_weight_index]
            child.weights[random_weight_index:] = parents[1].weights[random_weight_index:]
            child.biases[:random_bias_index] = parents[0].biases[:random_bias_index]
            child.biases[random_bias_index:] = parents[1].biases[random_bias_index:]
            
            return child
        else:
           return copy.deepcopy(parents[0])

    def update(self, agents):
        for i, agent in enumerate(agents):
            self.population[i].fitness = agent.fitness
    
    def upgrade(self):
        self.population[::-1].sort()
        
        # Initialize new population with top elites
        new_population = [self.population[0], self.population[1], self.population[2]]
        
        # Generate elite crossovers
        new_population.extend([
            self.crossover(self.population[0], self.population[1]),
            self.crossover(self.population[0], self.population[2]),
            self.crossover(self.population[1], self.population[2])
        ])				

        # Fill remaining slots with tournament-selected crossovers
        for _ in range(len(new_population), self.population_size):
            parents = (self.get_genome_by_tournament(),
                       self.get_genome_by_tournament())
            new_population.append(self.crossover(*parents))

        # Mutate all except the best genome
        for genome in new_population[1:]:
            genome.mutate()
                
        self.population = np.array(new_population)