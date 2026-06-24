import copy
import random
import numpy as np


class Genome():
    def __init__(self, weights, biases):
        self.fitness = 0
        self.weights = weights
        self.biases = biases

    def mutate(self):
        # Small random gene nudges make the next generation explore nearby
        # neural networks while keeping good solutions recognizable.
        mutation_rate = 0.18
        mutation_strength = 0.35
        for i in range(len(self.weights)):
            if np.random.random() < mutation_rate:
                self.weights[i] += np.random.normal(0, mutation_strength)

        for i in range(len(self.biases)):
            if np.random.random() < mutation_rate:
                self.biases[i] += np.random.normal(0, mutation_strength)

    def __lt__(self, other_genome):
        return self.fitness < other_genome.fitness


class Genetic_algorithm():
    def __init__(self, population_size, number_of_weights, number_of_biases):
        self.population_size = population_size
        self.population = [None] * population_size
        self.number_of_weights = number_of_weights
        self.number_of_biases = number_of_biases

        for i in range(population_size):
            initial_weights = np.random.uniform(-2, 2, number_of_weights).tolist()
            initial_biases = np.random.uniform(-2, 2, number_of_biases).tolist()
            self.population[i] = Genome(initial_weights, initial_biases)

    def get_genome_by_tournament(self):
        tournament_size = min(4, len(self.population))
        combatants = np.random.choice(range(len(self.population)), tournament_size, replace=False)
        fittest_genome = self.population[combatants[0]]
        for combatant in combatants[1:]:
            if self.population[combatant].fitness > fittest_genome.fitness:
                fittest_genome = self.population[combatant]
        return fittest_genome

    def crossover(self, parent_0, parent_1):
        crossover_rate = 0.95
        parents = np.random.choice([parent_0, parent_1], 2, replace=False)

        if np.random.random() < crossover_rate:
            random_weight_index = random.randint(0, self.number_of_weights)
            random_bias_index = random.randint(0, self.number_of_biases)
            child = Genome([None] * self.number_of_weights, [None] * self.number_of_biases)

            child.weights[0:random_weight_index] = parents[0].weights[0:random_weight_index]
            child.weights[random_weight_index:] = parents[1].weights[random_weight_index:]
            child.biases[0:random_bias_index] = parents[0].biases[0:random_bias_index]
            child.biases[random_bias_index:] = parents[1].biases[random_bias_index:]
            return child

        return copy.deepcopy(parents[0])

    def update(self, agents):
        for i, agent in enumerate(agents):
            self.population[i].fitness = agent.fitness

    def upgrade(self):
        self.population.sort(reverse=True)

        new_population = [None] * self.population_size
        elite_count = min(4, self.population_size)
        for i in range(elite_count):
            new_population[i] = copy.deepcopy(self.population[i])

        for i in range(elite_count, self.population_size):
            parent_0 = self.get_genome_by_tournament()
            parent_1 = self.get_genome_by_tournament()
            new_population[i] = self.crossover(parent_0, parent_1)
            new_population[i].mutate()

        self.population = new_population
