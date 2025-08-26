"""Bio-Inspired AI Module for AIBF Framework.

This module provides bio-inspired artificial intelligence capabilities including:
- Evolutionary algorithms
- Swarm intelligence
- Artificial immune systems
- Neural development and growth
- Biomimetic optimization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import random
import copy
from collections import defaultdict

logger = logging.getLogger(__name__)


class EvolutionStrategy(Enum):
    """Evolutionary algorithm strategies."""
    GENETIC_ALGORITHM = "GA"
    EVOLUTION_STRATEGY = "ES"
    GENETIC_PROGRAMMING = "GP"
    DIFFERENTIAL_EVOLUTION = "DE"
    PARTICLE_SWARM = "PSO"


class SelectionMethod(Enum):
    """Selection methods for evolutionary algorithms."""
    TOURNAMENT = "TOURNAMENT"
    ROULETTE_WHEEL = "ROULETTE"
    RANK_BASED = "RANK"
    ELITIST = "ELITIST"


class SwarmBehavior(Enum):
    """Swarm intelligence behaviors."""
    PARTICLE_SWARM = "PSO"
    ANT_COLONY = "ACO"
    BEE_COLONY = "ABC"
    FIREFLY = "FA"
    CUCKOO_SEARCH = "CS"


@dataclass
class Individual:
    """Represents an individual in evolutionary algorithms."""
    genome: List[float]
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.genome, list):
            self.genome = list(self.genome)


@dataclass
class Particle:
    """Represents a particle in swarm optimization."""
    position: List[float]
    velocity: List[float]
    best_position: List[float]
    best_fitness: float = float('inf')
    fitness: float = float('inf')
    
    def __post_init__(self):
        if self.best_position is None:
            self.best_position = self.position.copy()


class GeneticAlgorithm:
    """Genetic Algorithm implementation."""
    
    def __init__(self, population_size: int = 100, genome_length: int = 10,
                 mutation_rate: float = 0.01, crossover_rate: float = 0.8,
                 selection_method: SelectionMethod = SelectionMethod.TOURNAMENT):
        self.population_size = population_size
        self.genome_length = genome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        
    def initialize_population(self, bounds: Tuple[float, float] = (-1.0, 1.0)):
        """Initialize random population."""
        self.population = []
        for i in range(self.population_size):
            genome = [random.uniform(bounds[0], bounds[1]) for _ in range(self.genome_length)]
            individual = Individual(genome=genome, generation=self.generation)
            self.population.append(individual)
    
    def evaluate_population(self, fitness_function: Callable[[List[float]], float]):
        """Evaluate fitness for entire population."""
        for individual in self.population:
            individual.fitness = fitness_function(individual.genome)
            
        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(current_best)
            
        # Record fitness history
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        self.fitness_history.append(avg_fitness)
    
    def selection(self, num_parents: int) -> List[Individual]:
        """Select parents for reproduction."""
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(num_parents)
        elif self.selection_method == SelectionMethod.ROULETTE_WHEEL:
            return self._roulette_wheel_selection(num_parents)
        elif self.selection_method == SelectionMethod.RANK_BASED:
            return self._rank_based_selection(num_parents)
        else:
            return self._elitist_selection(num_parents)
    
    def _tournament_selection(self, num_parents: int, tournament_size: int = 3) -> List[Individual]:
        """Tournament selection."""
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(copy.deepcopy(winner))
        return parents
    
    def _roulette_wheel_selection(self, num_parents: int) -> List[Individual]:
        """Roulette wheel selection."""
        # Shift fitness to ensure all values are positive
        min_fitness = min(ind.fitness for ind in self.population)
        adjusted_fitness = [ind.fitness - min_fitness + 1e-6 for ind in self.population]
        total_fitness = sum(adjusted_fitness)
        
        parents = []
        for _ in range(num_parents):
            pick = random.uniform(0, total_fitness)
            current = 0
            for i, fitness in enumerate(adjusted_fitness):
                current += fitness
                if current >= pick:
                    parents.append(copy.deepcopy(self.population[i]))
                    break
        return parents
    
    def _rank_based_selection(self, num_parents: int) -> List[Individual]:
        """Rank-based selection."""
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        ranks = list(range(len(sorted_population), 0, -1))
        total_rank = sum(ranks)
        
        parents = []
        for _ in range(num_parents):
            pick = random.uniform(0, total_rank)
            current = 0
            for i, rank in enumerate(ranks):
                current += rank
                if current >= pick:
                    parents.append(copy.deepcopy(sorted_population[i]))
                    break
        return parents
    
    def _elitist_selection(self, num_parents: int) -> List[Individual]:
        """Elitist selection - select best individuals."""
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return [copy.deepcopy(ind) for ind in sorted_population[:num_parents]]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        crossover_point = random.randint(1, len(parent1.genome) - 1)
        
        child1_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
        child2_genome = parent2.genome[:crossover_point] + parent1.genome[crossover_point:]
        
        child1 = Individual(genome=child1_genome, generation=self.generation + 1)
        child2 = Individual(genome=child2_genome, generation=self.generation + 1)
        
        return child1, child2
    
    def mutate(self, individual: Individual, bounds: Tuple[float, float] = (-1.0, 1.0)):
        """Gaussian mutation."""
        for i in range(len(individual.genome)):
            if random.random() < self.mutation_rate:
                mutation = random.gauss(0, 0.1)
                individual.genome[i] += mutation
                # Ensure bounds
                individual.genome[i] = max(bounds[0], min(bounds[1], individual.genome[i]))
    
    def evolve_generation(self, fitness_function: Callable[[List[float]], float],
                         bounds: Tuple[float, float] = (-1.0, 1.0)) -> Dict[str, Any]:
        """Evolve one generation."""
        try:
            # Evaluate current population
            self.evaluate_population(fitness_function)
            
            # Select parents
            num_parents = self.population_size // 2
            parents = self.selection(num_parents)
            
            # Create new population
            new_population = []
            
            # Elitism - keep best individuals
            elite_size = max(1, self.population_size // 10)
            elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:elite_size]
            new_population.extend([copy.deepcopy(ind) for ind in elite])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutate(child1, bounds)
                self.mutate(child2, bounds)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            self.population = new_population[:self.population_size]
            self.generation += 1
            
            return {
                "generation": self.generation,
                "best_fitness": self.best_individual.fitness if self.best_individual else 0,
                "average_fitness": self.fitness_history[-1] if self.fitness_history else 0,
                "population_diversity": self._calculate_diversity()
            }
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            raise
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
            
        total_distance = 0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.linalg.norm(
                    np.array(self.population[i].genome) - np.array(self.population[j].genome)
                )
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0


class ParticleSwarmOptimization:
    """Particle Swarm Optimization implementation."""
    
    def __init__(self, num_particles: int = 30, dimensions: int = 10,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.particles: List[Particle] = []
        self.global_best_position: List[float] = []
        self.global_best_fitness: float = float('inf')
        self.iteration = 0
        self.fitness_history: List[float] = []
        
    def initialize_swarm(self, bounds: Tuple[float, float] = (-10.0, 10.0)):
        """Initialize particle swarm."""
        self.particles = []
        for _ in range(self.num_particles):
            position = [random.uniform(bounds[0], bounds[1]) for _ in range(self.dimensions)]
            velocity = [random.uniform(-1, 1) for _ in range(self.dimensions)]
            particle = Particle(position=position, velocity=velocity, best_position=position.copy())
            self.particles.append(particle)
    
    def evaluate_swarm(self, fitness_function: Callable[[List[float]], float]):
        """Evaluate fitness for all particles."""
        for particle in self.particles:
            particle.fitness = fitness_function(particle.position)
            
            # Update personal best
            if particle.fitness < particle.best_fitness:
                particle.best_fitness = particle.fitness
                particle.best_position = particle.position.copy()
            
            # Update global best
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
        
        # Record fitness history
        avg_fitness = sum(p.fitness for p in self.particles) / len(self.particles)
        self.fitness_history.append(avg_fitness)
    
    def update_velocities_and_positions(self, bounds: Tuple[float, float] = (-10.0, 10.0)):
        """Update particle velocities and positions."""
        for particle in self.particles:
            for d in range(self.dimensions):
                # Update velocity
                r1, r2 = random.random(), random.random()
                
                cognitive_component = self.c1 * r1 * (particle.best_position[d] - particle.position[d])
                social_component = self.c2 * r2 * (self.global_best_position[d] - particle.position[d])
                
                particle.velocity[d] = (self.w * particle.velocity[d] + 
                                      cognitive_component + social_component)
                
                # Limit velocity
                max_velocity = (bounds[1] - bounds[0]) * 0.1
                particle.velocity[d] = max(-max_velocity, min(max_velocity, particle.velocity[d]))
                
                # Update position
                particle.position[d] += particle.velocity[d]
                
                # Ensure bounds
                particle.position[d] = max(bounds[0], min(bounds[1], particle.position[d]))
    
    def optimize(self, fitness_function: Callable[[List[float]], float],
                max_iterations: int = 100, bounds: Tuple[float, float] = (-10.0, 10.0)) -> Dict[str, Any]:
        """Run PSO optimization."""
        try:
            self.initialize_swarm(bounds)
            
            for iteration in range(max_iterations):
                self.evaluate_swarm(fitness_function)
                self.update_velocities_and_positions(bounds)
                self.iteration = iteration + 1
                
                if iteration % 10 == 0:
                    logger.info(f"PSO Iteration {iteration}, Best Fitness: {self.global_best_fitness:.6f}")
            
            return {
                "best_position": self.global_best_position,
                "best_fitness": self.global_best_fitness,
                "iterations": max_iterations,
                "fitness_history": self.fitness_history,
                "convergence_iteration": self._find_convergence_point()
            }
            
        except Exception as e:
            logger.error(f"PSO optimization failed: {e}")
            raise
    
    def _find_convergence_point(self, tolerance: float = 1e-6) -> int:
        """Find iteration where algorithm converged."""
        if len(self.fitness_history) < 10:
            return len(self.fitness_history)
            
        for i in range(10, len(self.fitness_history)):
            recent_values = self.fitness_history[i-10:i]
            if max(recent_values) - min(recent_values) < tolerance:
                return i
        
        return len(self.fitness_history)


class AntColonyOptimization:
    """Ant Colony Optimization for combinatorial problems."""
    
    def __init__(self, num_ants: int = 20, alpha: float = 1.0, beta: float = 2.0,
                 rho: float = 0.1, q0: float = 0.9):
        self.num_ants = num_ants
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic importance
        self.rho = rho      # Evaporation rate
        self.q0 = q0        # Exploitation vs exploration
        self.pheromone_matrix: Optional[np.ndarray] = None
        self.distance_matrix: Optional[np.ndarray] = None
        self.best_tour: List[int] = []
        self.best_distance: float = float('inf')
        
    def solve_tsp(self, distance_matrix: np.ndarray, max_iterations: int = 100) -> Dict[str, Any]:
        """Solve Traveling Salesman Problem using ACO."""
        try:
            self.distance_matrix = distance_matrix
            num_cities = len(distance_matrix)
            
            # Initialize pheromone matrix
            self.pheromone_matrix = np.ones((num_cities, num_cities)) * 0.1
            
            iteration_best_distances = []
            
            for iteration in range(max_iterations):
                # Generate ant solutions
                ant_tours = []
                ant_distances = []
                
                for ant in range(self.num_ants):
                    tour = self._construct_tour(num_cities)
                    distance = self._calculate_tour_distance(tour)
                    
                    ant_tours.append(tour)
                    ant_distances.append(distance)
                    
                    # Update best solution
                    if distance < self.best_distance:
                        self.best_distance = distance
                        self.best_tour = tour.copy()
                
                # Update pheromones
                self._update_pheromones(ant_tours, ant_distances)
                
                iteration_best = min(ant_distances)
                iteration_best_distances.append(iteration_best)
                
                if iteration % 10 == 0:
                    logger.info(f"ACO Iteration {iteration}, Best Distance: {self.best_distance:.2f}")
            
            return {
                "best_tour": self.best_tour,
                "best_distance": self.best_distance,
                "iterations": max_iterations,
                "convergence_history": iteration_best_distances
            }
            
        except Exception as e:
            logger.error(f"ACO TSP solving failed: {e}")
            raise
    
    def _construct_tour(self, num_cities: int) -> List[int]:
        """Construct tour for single ant."""
        tour = [0]  # Start from city 0
        unvisited = set(range(1, num_cities))
        
        while unvisited:
            current_city = tour[-1]
            next_city = self._select_next_city(current_city, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
        
        return tour
    
    def _select_next_city(self, current_city: int, unvisited: set) -> int:
        """Select next city using ACO probability rules."""
        if random.random() < self.q0:
            # Exploitation: choose best option
            best_city = None
            best_value = -1
            
            for city in unvisited:
                pheromone = self.pheromone_matrix[current_city][city]
                heuristic = 1.0 / (self.distance_matrix[current_city][city] + 1e-10)
                value = (pheromone ** self.alpha) * (heuristic ** self.beta)
                
                if value > best_value:
                    best_value = value
                    best_city = city
            
            return best_city
        else:
            # Exploration: probabilistic selection
            probabilities = []
            total = 0
            
            for city in unvisited:
                pheromone = self.pheromone_matrix[current_city][city]
                heuristic = 1.0 / (self.distance_matrix[current_city][city] + 1e-10)
                prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append(prob)
                total += prob
            
            # Normalize probabilities
            probabilities = [p / total for p in probabilities]
            
            # Select city based on probabilities
            r = random.random()
            cumulative = 0
            for i, city in enumerate(unvisited):
                cumulative += probabilities[i]
                if r <= cumulative:
                    return city
            
            return list(unvisited)[0]  # Fallback
    
    def _calculate_tour_distance(self, tour: List[int]) -> float:
        """Calculate total distance of tour."""
        distance = 0
        for i in range(len(tour)):
            from_city = tour[i]
            to_city = tour[(i + 1) % len(tour)]  # Return to start
            distance += self.distance_matrix[from_city][to_city]
        return distance
    
    def _update_pheromones(self, ant_tours: List[List[int]], ant_distances: List[float]):
        """Update pheromone matrix."""
        # Evaporation
        self.pheromone_matrix *= (1 - self.rho)
        
        # Deposit pheromones
        for tour, distance in zip(ant_tours, ant_distances):
            pheromone_deposit = 1.0 / distance
            
            for i in range(len(tour)):
                from_city = tour[i]
                to_city = tour[(i + 1) % len(tour)]
                self.pheromone_matrix[from_city][to_city] += pheromone_deposit
                self.pheromone_matrix[to_city][from_city] += pheromone_deposit  # Symmetric


class ArtificialImmuneSystem:
    """Artificial Immune System for anomaly detection and optimization."""
    
    def __init__(self, num_antibodies: int = 50, clone_factor: int = 10,
                 mutation_rate: float = 0.1, selection_pressure: float = 0.2):
        self.num_antibodies = num_antibodies
        self.clone_factor = clone_factor
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.antibodies: List[Individual] = []
        self.memory_cells: List[Individual] = []
        
    def initialize_antibodies(self, dimensions: int, bounds: Tuple[float, float] = (-1.0, 1.0)):
        """Initialize antibody population."""
        self.antibodies = []
        for _ in range(self.num_antibodies):
            genome = [random.uniform(bounds[0], bounds[1]) for _ in range(dimensions)]
            antibody = Individual(genome=genome)
            self.antibodies.append(antibody)
    
    def clonal_selection(self, antigen_function: Callable[[List[float]], float],
                        max_generations: int = 100) -> Dict[str, Any]:
        """Perform clonal selection algorithm."""
        try:
            best_fitness_history = []
            
            for generation in range(max_generations):
                # Evaluate antibodies
                for antibody in self.antibodies:
                    antibody.fitness = antigen_function(antibody.genome)
                
                # Sort by fitness (higher is better)
                self.antibodies.sort(key=lambda x: x.fitness, reverse=True)
                
                # Select best antibodies for cloning
                num_selected = int(len(self.antibodies) * self.selection_pressure)
                selected = self.antibodies[:num_selected]
                
                # Clone and mutate
                new_antibodies = []
                for antibody in selected:
                    # Number of clones proportional to fitness
                    num_clones = max(1, int(self.clone_factor * antibody.fitness / 
                                          (max(ab.fitness for ab in self.antibodies) + 1e-10)))
                    
                    for _ in range(num_clones):
                        clone = copy.deepcopy(antibody)
                        self._hypermutate(clone)
                        new_antibodies.append(clone)
                
                # Replace worst antibodies with new clones
                num_replace = min(len(new_antibodies), len(self.antibodies) - num_selected)
                self.antibodies = selected + new_antibodies[:num_replace]
                
                # Update memory cells
                self._update_memory()
                
                best_fitness = max(ab.fitness for ab in self.antibodies)
                best_fitness_history.append(best_fitness)
                
                if generation % 10 == 0:
                    logger.info(f"AIS Generation {generation}, Best Fitness: {best_fitness:.6f}")
            
            best_antibody = max(self.antibodies, key=lambda x: x.fitness)
            
            return {
                "best_solution": best_antibody.genome,
                "best_fitness": best_antibody.fitness,
                "generations": max_generations,
                "fitness_history": best_fitness_history,
                "memory_cells": len(self.memory_cells)
            }
            
        except Exception as e:
            logger.error(f"Clonal selection failed: {e}")
            raise
    
    def _hypermutate(self, antibody: Individual):
        """Apply hypermutation to antibody."""
        # Mutation rate inversely proportional to fitness
        max_fitness = max(ab.fitness for ab in self.antibodies) if self.antibodies else 1.0
        adaptive_rate = self.mutation_rate * (1.0 - antibody.fitness / (max_fitness + 1e-10))
        
        for i in range(len(antibody.genome)):
            if random.random() < adaptive_rate:
                antibody.genome[i] += random.gauss(0, 0.1)
                antibody.genome[i] = max(-1.0, min(1.0, antibody.genome[i]))  # Bounds
    
    def _update_memory(self):
        """Update memory cells with best antibodies."""
        # Add best antibodies to memory
        best_antibodies = sorted(self.antibodies, key=lambda x: x.fitness, reverse=True)[:5]
        
        for antibody in best_antibodies:
            # Check if similar antibody already in memory
            is_novel = True
            for memory_cell in self.memory_cells:
                similarity = self._calculate_similarity(antibody.genome, memory_cell.genome)
                if similarity > 0.9:  # Too similar
                    is_novel = False
                    break
            
            if is_novel:
                self.memory_cells.append(copy.deepcopy(antibody))
        
        # Limit memory size
        if len(self.memory_cells) > 20:
            self.memory_cells = sorted(self.memory_cells, key=lambda x: x.fitness, reverse=True)[:20]
    
    def _calculate_similarity(self, genome1: List[float], genome2: List[float]) -> float:
        """Calculate similarity between two genomes."""
        if len(genome1) != len(genome2):
            return 0.0
        
        distance = np.linalg.norm(np.array(genome1) - np.array(genome2))
        max_distance = np.sqrt(len(genome1) * 4)  # Assuming bounds [-1, 1]
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, similarity)


class BioInspiredManager:
    """Main interface for bio-inspired AI capabilities."""
    
    def __init__(self):
        self.algorithms: Dict[str, Any] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
    def create_genetic_algorithm(self, name: str, **params) -> GeneticAlgorithm:
        """Create genetic algorithm instance."""
        ga = GeneticAlgorithm(**params)
        self.algorithms[name] = ga
        return ga
    
    def create_particle_swarm(self, name: str, **params) -> ParticleSwarmOptimization:
        """Create particle swarm optimization instance."""
        pso = ParticleSwarmOptimization(**params)
        self.algorithms[name] = pso
        return pso
    
    def create_ant_colony(self, name: str, **params) -> AntColonyOptimization:
        """Create ant colony optimization instance."""
        aco = AntColonyOptimization(**params)
        self.algorithms[name] = aco
        return aco
    
    def create_immune_system(self, name: str, **params) -> ArtificialImmuneSystem:
        """Create artificial immune system instance."""
        ais = ArtificialImmuneSystem(**params)
        self.algorithms[name] = ais
        return ais
    
    def optimize_function(self, algorithm_name: str, fitness_function: Callable,
                         strategy: EvolutionStrategy, **kwargs) -> Dict[str, Any]:
        """Optimize function using specified bio-inspired algorithm."""
        try:
            if algorithm_name not in self.algorithms:
                raise ValueError(f"Algorithm '{algorithm_name}' not found")
            
            algorithm = self.algorithms[algorithm_name]
            
            if strategy == EvolutionStrategy.GENETIC_ALGORITHM:
                if not isinstance(algorithm, GeneticAlgorithm):
                    raise ValueError("Algorithm must be GeneticAlgorithm for GA strategy")
                
                # Run GA optimization
                max_generations = kwargs.get('max_generations', 100)
                bounds = kwargs.get('bounds', (-1.0, 1.0))
                
                algorithm.initialize_population(bounds)
                
                for generation in range(max_generations):
                    result = algorithm.evolve_generation(fitness_function, bounds)
                    if generation % 10 == 0:
                        logger.info(f"GA Generation {generation}: {result}")
                
                final_result = {
                    "algorithm": "Genetic Algorithm",
                    "best_solution": algorithm.best_individual.genome if algorithm.best_individual else None,
                    "best_fitness": algorithm.best_individual.fitness if algorithm.best_individual else 0,
                    "generations": max_generations,
                    "fitness_history": algorithm.fitness_history
                }
                
            elif strategy == EvolutionStrategy.PARTICLE_SWARM:
                if not isinstance(algorithm, ParticleSwarmOptimization):
                    raise ValueError("Algorithm must be ParticleSwarmOptimization for PSO strategy")
                
                max_iterations = kwargs.get('max_iterations', 100)
                bounds = kwargs.get('bounds', (-10.0, 10.0))
                
                final_result = algorithm.optimize(fitness_function, max_iterations, bounds)
                final_result["algorithm"] = "Particle Swarm Optimization"
                
            else:
                raise NotImplementedError(f"Strategy {strategy.value} not implemented")
            
            self.optimization_history.append(final_result)
            return final_result
            
        except Exception as e:
            logger.error(f"Bio-inspired optimization failed: {e}")
            raise
    
    def solve_combinatorial_problem(self, algorithm_name: str, problem_data: Any,
                                   problem_type: str = "TSP", **kwargs) -> Dict[str, Any]:
        """Solve combinatorial optimization problems."""
        try:
            if algorithm_name not in self.algorithms:
                raise ValueError(f"Algorithm '{algorithm_name}' not found")
            
            algorithm = self.algorithms[algorithm_name]
            
            if problem_type == "TSP":
                if not isinstance(algorithm, AntColonyOptimization):
                    raise ValueError("Algorithm must be AntColonyOptimization for TSP")
                
                max_iterations = kwargs.get('max_iterations', 100)
                result = algorithm.solve_tsp(problem_data, max_iterations)
                result["algorithm"] = "Ant Colony Optimization"
                result["problem_type"] = "TSP"
                
                self.optimization_history.append(result)
                return result
            else:
                raise NotImplementedError(f"Problem type {problem_type} not implemented")
                
        except Exception as e:
            logger.error(f"Combinatorial problem solving failed: {e}")
            raise
    
    def get_algorithm_info(self, algorithm_name: str) -> Dict[str, Any]:
        """Get information about algorithm."""
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Algorithm '{algorithm_name}' not found")
        
        algorithm = self.algorithms[algorithm_name]
        
        info = {
            "name": algorithm_name,
            "type": type(algorithm).__name__,
            "parameters": {}
        }
        
        if isinstance(algorithm, GeneticAlgorithm):
            info["parameters"] = {
                "population_size": algorithm.population_size,
                "genome_length": algorithm.genome_length,
                "mutation_rate": algorithm.mutation_rate,
                "crossover_rate": algorithm.crossover_rate,
                "generation": algorithm.generation
            }
        elif isinstance(algorithm, ParticleSwarmOptimization):
            info["parameters"] = {
                "num_particles": algorithm.num_particles,
                "dimensions": algorithm.dimensions,
                "inertia_weight": algorithm.w,
                "cognitive_param": algorithm.c1,
                "social_param": algorithm.c2,
                "iteration": algorithm.iteration
            }
        
        return info
    
    def get_bio_inspired_info(self) -> Dict[str, Any]:
        """Get information about bio-inspired AI capabilities."""
        return {
            "supported_algorithms": [
                "Genetic Algorithm",
                "Particle Swarm Optimization", 
                "Ant Colony Optimization",
                "Artificial Immune System"
            ],
            "evolution_strategies": [strategy.value for strategy in EvolutionStrategy],
            "selection_methods": [method.value for method in SelectionMethod],
            "swarm_behaviors": [behavior.value for behavior in SwarmBehavior],
            "active_algorithms": list(self.algorithms.keys()),
            "optimization_runs": len(self.optimization_history)
        }


# Utility functions
def sphere_function(x: List[float]) -> float:
    """Sphere function for testing optimization algorithms."""
    return -sum(xi**2 for xi in x)  # Negative for maximization


def rastrigin_function(x: List[float]) -> float:
    """Rastrigin function for testing optimization algorithms."""
    A = 10
    n = len(x)
    return -(A * n + sum(xi**2 - A * np.cos(2 * np.pi * xi) for xi in x))


def create_random_tsp_matrix(num_cities: int) -> np.ndarray:
    """Create random distance matrix for TSP testing."""
    matrix = np.random.uniform(1, 100, (num_cities, num_cities))
    # Make symmetric
    matrix = (matrix + matrix.T) / 2
    # Zero diagonal
    np.fill_diagonal(matrix, 0)
    return matrix


# Export main classes and functions
__all__ = [
    'BioInspiredManager',
    'GeneticAlgorithm',
    'ParticleSwarmOptimization',
    'AntColonyOptimization',
    'ArtificialImmuneSystem',
    'Individual',
    'Particle',
    'EvolutionStrategy',
    'SelectionMethod',
    'SwarmBehavior',
    'sphere_function',
    'rastrigin_function',
    'create_random_tsp_matrix'
]