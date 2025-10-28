import numpy as np
import random
import copy
from typing import Dict, List, Tuple
import json
from training_and_evaluation import train_model

class GeneticAlgorithm:
    """Genetic Algorithm for hyperparameter optimization"""
    
    def __init__(self, data_path: str, population_size: int = 10, 
                 generations: int = 5, epochs_per_individual: int = 5):
        self.data_path = data_path
        self.population_size = population_size
        self.generations = generations
        self.epochs_per_individual = epochs_per_individual
        
        # Define search space
        self.search_space = {
            'learning_rate': (1e-5, 1e-2),
            'batch_size': [16, 32, 64],
            'num_filters': [16, 32, 64],
            'dropout_rate': (0.1, 0.7),
            'weight_decay': (1e-6, 1e-3),
            'augmentation': {
                'horizontal_flip': [True, False],
                'rotation': [0, 10, 20, 30],
                'brightness': [0, 0.1, 0.2, 0.3],
                'contrast': [0, 0.1, 0.2, 0.3]
            }
        }
        
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_fitness = 0.0
        self.history = []
    
    def create_individual(self) -> Dict:
        """Create a random individual (set of hyperparameters)"""
        individual = {
            'learning_rate': np.random.uniform(*self.search_space['learning_rate']),
            'batch_size': random.choice(self.search_space['batch_size']),
            'num_filters': random.choice(self.search_space['num_filters']),
            'dropout_rate': np.random.uniform(*self.search_space['dropout_rate']),
            'weight_decay': np.random.uniform(*self.search_space['weight_decay']),
            'augmentation': {
                'horizontal_flip': random.choice(self.search_space['augmentation']['horizontal_flip']),
                'rotation': random.choice(self.search_space['augmentation']['rotation']),
                'brightness': random.choice(self.search_space['augmentation']['brightness']),
                'contrast': random.choice(self.search_space['augmentation']['contrast'])
            }
        }
        return individual
    
    def initialize_population(self):
        """Initialize population with random individuals"""
        self.population = [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, individual: Dict) -> float:
        """Evaluate fitness of an individual"""
        print(f"\nEvaluating individual:")
        print(f"  LR: {individual['learning_rate']:.6f}, Batch: {individual['batch_size']}, "
              f"Filters: {individual['num_filters']}, Dropout: {individual['dropout_rate']:.3f}")
        
        try:
            model, val_acc, f1 = train_model(individual, self.data_path, 
                                           epochs=self.epochs_per_individual)
            # Fitness is combination of accuracy and F1 score
            fitness = 0.7 * val_acc + 0.3 * (f1 * 100)
            print(f"  Fitness: {fitness:.2f} (Val Acc: {val_acc:.2f}%, F1: {f1:.4f})")
            return fitness
        except Exception as e:
            print(f"  Error during training: {e}")
            return 0.0
    
    def selection(self) -> List[Dict]:
        """Select parents using tournament selection"""
        parents = []
        for _ in range(self.population_size // 2):
            # Tournament selection
            tournament = random.sample(list(zip(self.population, self.fitness_scores)), 3)
            winner = max(tournament, key=lambda x: x[1])[0]
            parents.append(winner)
        return parents
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Create two children from two parents"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Crossover numeric hyperparameters
        for key in ['learning_rate', 'dropout_rate', 'weight_decay']:
            if random.random() < 0.5:
                child1[key], child2[key] = child2[key], child1[key]
        
        # Crossover categorical hyperparameters
        for key in ['batch_size', 'num_filters']:
            if random.random() < 0.5:
                child1[key], child2[key] = child2[key], child1[key]
        
        # Crossover augmentation parameters
        for key in child1['augmentation'].keys():
            if random.random() < 0.5:
                child1['augmentation'][key], child2['augmentation'][key] = \
                    child2['augmentation'][key], child1['augmentation'][key]
        
        return child1, child2
    
    def mutate(self, individual: Dict, mutation_rate: float = 0.2):
        """Mutate an individual"""
        mutated = copy.deepcopy(individual)
        
        # Mutate numeric hyperparameters
        if random.random() < mutation_rate:
            mutated['learning_rate'] = np.random.uniform(*self.search_space['learning_rate'])
        
        if random.random() < mutation_rate:
            mutated['dropout_rate'] = np.random.uniform(*self.search_space['dropout_rate'])
        
        if random.random() < mutation_rate:
            mutated['weight_decay'] = np.random.uniform(*self.search_space['weight_decay'])
        
        # Mutate categorical hyperparameters
        if random.random() < mutation_rate:
            mutated['batch_size'] = random.choice(self.search_space['batch_size'])
        
        if random.random() < mutation_rate:
            mutated['num_filters'] = random.choice(self.search_space['num_filters'])
        
        # Mutate augmentation parameters
        for key in mutated['augmentation'].keys():
            if random.random() < mutation_rate:
                mutated['augmentation'][key] = random.choice(
                    self.search_space['augmentation'][key]
                )
        
        return mutated
    
    def evolve(self):
        """Run the genetic algorithm"""
        print(f"\n{'='*80}")
        print(f"Starting Genetic Algorithm Optimization")
        print(f"Population Size: {self.population_size}, Generations: {self.generations}")
        print(f"{'='*80}\n")
        
        # Initialize population
        self.initialize_population()
        
        for generation in range(self.generations):
            print(f"\n{'='*80}")
            print(f"Generation {generation + 1}/{self.generations}")
            print(f"{'='*80}")
            
            # Evaluate fitness
            self.fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
            
            # Track best individual
            max_fitness_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[max_fitness_idx] > self.best_fitness:
                self.best_fitness = self.fitness_scores[max_fitness_idx]
                self.best_individual = copy.deepcopy(self.population[max_fitness_idx])
            
            # Store generation statistics
            gen_stats = {
                'generation': generation + 1,
                'best_fitness': float(np.max(self.fitness_scores)),
                'avg_fitness': float(np.mean(self.fitness_scores)),
                'std_fitness': float(np.std(self.fitness_scores))
            }
            self.history.append(gen_stats)
            
            print(f"\nGeneration {generation + 1} Summary:")
            print(f"  Best Fitness: {gen_stats['best_fitness']:.2f}")
            print(f"  Avg Fitness: {gen_stats['avg_fitness']:.2f}")
            print(f"  Std Fitness: {gen_stats['std_fitness']:.2f}")
            
            if generation < self.generations - 1:
                # Selection
                parents = self.selection()
                
                # Create new population
                new_population = []
                
                # Elitism: keep best individual
                new_population.append(copy.deepcopy(self.population[max_fitness_idx]))
                
                # Crossover and mutation
                while len(new_population) < self.population_size:
                    parent1, parent2 = random.sample(parents, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    
                    new_population.append(child1)
                    if len(new_population) < self.population_size:
                        new_population.append(child2)
                
                self.population = new_population
        
        print(f"\n{'='*80}")
        print(f"Optimization Complete!")
        print(f"Best Fitness: {self.best_fitness:.2f}")
        print(f"Best Hyperparameters:")
        print(json.dumps(self.best_individual, indent=2))
        print(f"{'='*80}\n")
        
        return self.best_individual, self.best_fitness