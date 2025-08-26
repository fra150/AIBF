# Emerging Technologies API Reference

## Panoramica

Il modulo Emerging Technologies di AIBF implementa tecnologie all'avanguardia come Quantum Computing, Neuromorphic Computing, Bio-Inspired AI, Edge Computing e Federated Learning.

## Quantum Computing

### Quantum Neural Networks

```python
from src.emerging.quantum_computing.quantum_neural_networks import (
    QuantumNeuralNetwork, QuantumLayer, QuantumCircuit, QuantumOptimizer
)
import numpy as np
from qiskit import QuantumCircuit as QiskitCircuit

# Rete neurale quantistica
qnn = QuantumNeuralNetwork(
    num_qubits=4,
    num_layers=3,
    entanglement_strategy="linear",  # "linear", "circular", "full"
    measurement_basis="z",  # "z", "x", "y", "bell"
    backend="qasm_simulator"  # "qasm_simulator", "statevector_simulator", "ibmq_qasm_simulator"
)

# Inizializza circuito quantistico
quantum_circuit = QuantumCircuit(
    num_qubits=4,
    circuit_depth=6
)

# Aggiungi layer quantistici
for i in range(3):
    quantum_layer = QuantumLayer(
        num_qubits=4,
        rotation_gates=["rx", "ry", "rz"],
        entangling_gates=["cnot", "cz"],
        parameter_sharing=True
    )
    quantum_circuit.add_layer(quantum_layer)

print(f"Quantum circuit created with {quantum_circuit.num_parameters} parameters")
print(f"Circuit depth: {quantum_circuit.depth}")
print(f"Gate count: {quantum_circuit.gate_count}")

# Compila circuito per backend specifico
compiled_circuit = await qnn.compile_circuit(
    quantum_circuit,
    optimization_level=3,
    basis_gates=["u1", "u2", "u3", "cx"]
)

print(f"Compiled circuit depth: {compiled_circuit.depth}")
print(f"Gate count after optimization: {compiled_circuit.gate_count}")

# Training quantistico
quantum_optimizer = QuantumOptimizer(
    method="spsa",  # "spsa", "cobyla", "l_bfgs_b", "gradient_descent"
    learning_rate=0.01,
    max_iterations=1000,
    tolerance=1e-6
)

# Dati di training (esempio classificazione binaria)
X_train = np.random.randn(100, 4)  # 100 campioni, 4 features
y_train = np.random.randint(0, 2, 100)  # Labels binari

# Encoding classico-quantistico
encoded_data = await qnn.encode_classical_data(
    X_train,
    encoding_method="amplitude",  # "amplitude", "angle", "basis"
    normalization=True
)

print(f"Data encoded into quantum states: {encoded_data.shape}")

# Training loop
training_history = []
for epoch in range(50):
    epoch_loss = 0
    
    for batch_start in range(0, len(X_train), 32):  # Batch size 32
        batch_end = min(batch_start + 32, len(X_train))
        X_batch = encoded_data[batch_start:batch_end]
        y_batch = y_train[batch_start:batch_end]
        
        # Forward pass quantistico
        quantum_outputs = await qnn.forward(
            X_batch,
            shots=1024  # Numero di misurazioni
        )
        
        # Calcola loss
        loss = await qnn.compute_loss(
            quantum_outputs,
            y_batch,
            loss_function="cross_entropy"
        )
        
        # Backward pass (gradient estimation)
        gradients = await quantum_optimizer.estimate_gradients(
            qnn,
            X_batch,
            y_batch,
            parameter_shift_rule=True
        )
        
        # Aggiorna parametri
        await quantum_optimizer.update_parameters(
            qnn.parameters,
            gradients
        )
        
        epoch_loss += loss
    
    avg_loss = epoch_loss / (len(X_train) // 32)
    training_history.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

# Valutazione
X_test = np.random.randn(20, 4)
y_test = np.random.randint(0, 2, 20)

encoded_test = await qnn.encode_classical_data(X_test, encoding_method="amplitude")
predictions = await qnn.predict(encoded_test, shots=1024)

accuracy = np.mean((predictions > 0.5) == y_test)
print(f"Quantum NN Accuracy: {accuracy:.3f}")

# Analisi entanglement
entanglement_analysis = await qnn.analyze_entanglement(
    quantum_circuit,
    measure="concurrence"  # "concurrence", "negativity", "entropy"
)

print(f"Average entanglement: {entanglement_analysis.average_entanglement:.3f}")
print(f"Max entanglement: {entanglement_analysis.max_entanglement:.3f}")
print(f"Entanglement distribution: {entanglement_analysis.distribution}")
```

### Quantum Algorithms

```python
from src.emerging.quantum_computing.quantum_algorithms import (
    QAOA, VQE, QuantumSupremacyBenchmark, QuantumMachineLearning
)

# Quantum Approximate Optimization Algorithm (QAOA)
qaoa = QAOA(
    num_qubits=6,
    num_layers=3,
    problem_type="max_cut",  # "max_cut", "tsp", "portfolio_optimization"
    mixer_hamiltonian="x_rotation"
)

# Definisci problema di ottimizzazione (Max-Cut)
graph_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4)]
max_cut_problem = await qaoa.define_max_cut_problem(graph_edges)

print(f"Max-Cut problem defined with {len(graph_edges)} edges")
print(f"Problem Hamiltonian: {max_cut_problem.hamiltonian}")

# Ottimizzazione QAOA
optimization_result = await qaoa.optimize(
    problem=max_cut_problem,
    optimizer="cobyla",
    max_iterations=200,
    initial_parameters="random"
)

print(f"QAOA Optimization Results:")
print(f"  Optimal parameters: {optimization_result.optimal_parameters}")
print(f"  Optimal value: {optimization_result.optimal_value:.4f}")
print(f"  Convergence: {optimization_result.converged}")
print(f"  Iterations: {optimization_result.num_iterations}")

# Estrai soluzione
optimal_solution = await qaoa.extract_solution(
    optimization_result,
    num_samples=1000
)

print(f"Optimal cut: {optimal_solution.cut_edges}")
print(f"Cut value: {optimal_solution.cut_value}")
print(f"Solution probability: {optimal_solution.probability:.3f}")

# Variational Quantum Eigensolver (VQE)
vqe = VQE(
    num_qubits=4,
    ansatz="uccsd",  # "uccsd", "hardware_efficient", "real_amplitudes"
    optimizer="spsa",
    backend="statevector_simulator"
)

# Definisci Hamiltoniano molecolare (H2)
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver

# Molecola H2
molecule = PySCFDriver(
    atom="H .0 .0 .0; H .0 .0 0.735",
    unit="Angstrom",
    charge=0,
    spin=0,
    basis="sto3g"
)

molecular_hamiltonian = await vqe.prepare_molecular_hamiltonian(molecule)

print(f"Molecular Hamiltonian prepared:")
print(f"  Number of qubits: {molecular_hamiltonian.num_qubits}")
print(f"  Number of terms: {len(molecular_hamiltonian.terms)}")

# Calcola energia dello stato fondamentale
ground_state_result = await vqe.find_ground_state(
    hamiltonian=molecular_hamiltonian,
    initial_state="hf",  # Hartree-Fock
    max_iterations=500
)

print(f"VQE Ground State Results:")
print(f"  Ground state energy: {ground_state_result.eigenvalue:.6f} Ha")
print(f"  Optimal parameters: {ground_state_result.optimal_parameters}")
print(f"  Convergence: {ground_state_result.converged}")

# Quantum Machine Learning
qml = QuantumMachineLearning(
    algorithm="qsvm",  # "qsvm", "qnn", "qgan", "qrl"
    feature_map="zz_feature_map",
    num_qubits=4
)

# Quantum Support Vector Machine
X_qsvm = np.random.randn(50, 2)  # 2D dataset
y_qsvm = np.random.randint(0, 2, 50)

# Training QSVM
qsvm_model = await qml.train_qsvm(
    X_train=X_qsvm,
    y_train=y_qsvm,
    feature_map_depth=2,
    shots=1024
)

print(f"QSVM trained with {len(qsvm_model.support_vectors)} support vectors")

# Predizione
X_test_qsvm = np.random.randn(10, 2)
qsvm_predictions = await qml.predict_qsvm(qsvm_model, X_test_qsvm)

print(f"QSVM predictions: {qsvm_predictions}")

# Quantum Supremacy Benchmark
supremacy_benchmark = QuantumSupremacyBenchmark(
    num_qubits=20,
    circuit_depth=20,
    gate_set=["h", "cx", "rz"],
    random_seed=42
)

# Genera circuito random
random_circuit = await supremacy_benchmark.generate_random_circuit(
    gate_density=0.8,
    entanglement_probability=0.3
)

print(f"Random circuit generated:")
print(f"  Depth: {random_circuit.depth}")
print(f"  Gate count: {random_circuit.gate_count}")
print(f"  Two-qubit gates: {random_circuit.two_qubit_gate_count}")

# Simula circuito
simulation_result = await supremacy_benchmark.simulate_circuit(
    random_circuit,
    shots=1000,
    noise_model="ibmq_16_melbourne"  # Modello di rumore realistico
)

print(f"Simulation completed:")
print(f"  Execution time: {simulation_result.execution_time:.2f}s")
print(f"  Fidelity: {simulation_result.fidelity:.4f}")
print(f"  Cross-entropy benchmarking: {simulation_result.xeb_score:.4f}")
```

## Neuromorphic Computing

### Spiking Neural Networks

```python
from src.emerging.neuromorphic_computing.spiking_neural_networks import (
    SpikingNeuralNetwork, LIFNeuron, STDPLearning, SpikingLayer
)
import numpy as np

# Neurone Leaky Integrate-and-Fire
lif_neuron = LIFNeuron(
    threshold=1.0,
    reset_potential=0.0,
    membrane_resistance=10.0,  # MΩ
    membrane_capacitance=1.0,  # nF
    refractory_period=2.0,     # ms
    leak_conductance=0.1       # μS
)

# Layer di neuroni spiking
spiking_layer = SpikingLayer(
    num_neurons=100,
    neuron_type="lif",  # "lif", "izhikevich", "hodgkin_huxley"
    connection_probability=0.1,
    synaptic_delay_range=(1, 5),  # ms
    initial_weights="random"
)

# Rete neurale spiking
snn = SpikingNeuralNetwork(
    layers=[
        SpikingLayer(784, neuron_type="input"),     # Input layer (28x28 image)
        SpikingLayer(128, neuron_type="lif"),       # Hidden layer
        SpikingLayer(64, neuron_type="lif"),        # Hidden layer
        SpikingLayer(10, neuron_type="output")      # Output layer
    ],
    simulation_timestep=0.1,  # ms
    simulation_duration=100.0  # ms
)

# Inizializza rete
await snn.initialize(
    weight_initialization="xavier",
    bias_initialization="zero"
)

print(f"Spiking Neural Network initialized:")
print(f"  Total neurons: {snn.total_neurons}")
print(f"  Total synapses: {snn.total_synapses}")
print(f"  Network topology: {snn.topology}")

# Apprendimento STDP (Spike-Timing Dependent Plasticity)
stdp_learning = STDPLearning(
    learning_rate=0.01,
    tau_pre=20.0,   # ms - costante di tempo pre-sinaptica
    tau_post=20.0,  # ms - costante di tempo post-sinaptica
    a_plus=0.1,     # Ampiezza potenziamento
    a_minus=0.12,   # Ampiezza depressione
    weight_bounds=(0.0, 1.0)
)

# Codifica input in spike trains
def encode_image_to_spikes(image, encoding_method="rate", duration=100.0):
    """Codifica immagine in treni di spike"""
    if encoding_method == "rate":
        # Rate coding: intensità pixel -> frequenza spike
        spike_rates = image.flatten() * 100  # Hz
        spike_trains = []
        
        for rate in spike_rates:
            if rate > 0:
                # Genera spike times con distribuzione di Poisson
                num_spikes = np.random.poisson(rate * duration / 1000)
                spike_times = np.sort(np.random.uniform(0, duration, num_spikes))
            else:
                spike_times = np.array([])
            spike_trains.append(spike_times)
        
        return spike_trains
    
    elif encoding_method == "temporal":
        # Temporal coding: intensità pixel -> timing primo spike
        spike_trains = []
        for pixel in image.flatten():
            if pixel > 0.1:  # Soglia minima
                # Primo spike inversamente proporzionale all'intensità
                first_spike_time = (1.0 - pixel) * duration * 0.8
                spike_times = np.array([first_spike_time])
            else:
                spike_times = np.array([])
            spike_trains.append(spike_times)
        
        return spike_trains

# Simula training con dataset MNIST spiking
training_data = []  # Carica dataset MNIST
for epoch in range(10):
    epoch_accuracy = 0
    
    for image, label in training_data[:100]:  # Primi 100 campioni
        # Codifica immagine in spike trains
        input_spikes = encode_image_to_spikes(
            image,
            encoding_method="rate",
            duration=100.0
        )
        
        # Forward pass
        output_spikes = await snn.forward(
            input_spikes,
            simulation_duration=100.0,
            record_membrane_potential=True
        )
        
        # Decodifica output (spike count)
        output_counts = [len(spikes) for spikes in output_spikes]
        predicted_class = np.argmax(output_counts)
        
        # Calcola accuratezza
        if predicted_class == label:
            epoch_accuracy += 1
        
        # Apprendimento STDP
        if predicted_class != label:
            # Applica STDP solo se predizione errata
            await stdp_learning.update_weights(
                snn,
                input_spikes,
                output_spikes,
                target_class=label
            )
    
    epoch_accuracy /= 100
    print(f"Epoch {epoch + 1}: Accuracy = {epoch_accuracy:.3f}")

# Analisi attività di rete
network_activity = await snn.analyze_network_activity(
    input_spikes,
    metrics=["firing_rate", "synchrony", "burst_detection", "connectivity"]
)

print(f"\nNetwork Activity Analysis:")
print(f"  Average firing rate: {network_activity.avg_firing_rate:.2f} Hz")
print(f"  Network synchrony: {network_activity.synchrony_index:.3f}")
print(f"  Burst events detected: {network_activity.burst_count}")
print(f"  Effective connectivity: {network_activity.effective_connectivity:.3f}")

# Visualizzazione raster plot
raster_data = await snn.get_raster_plot_data()
print(f"Raster plot data: {len(raster_data.spike_times)} neurons, "
      f"{sum(len(spikes) for spikes in raster_data.spike_times)} total spikes")
```

### Neuromorphic Hardware Simulation

```python
from src.emerging.neuromorphic_computing.neuromorphic_hardware import (
    LoihiSimulator, SpiNNakerSimulator, TrueNorthSimulator, NeuromorphicChip
)

# Simulatore Intel Loihi
loihi_sim = LoihiSimulator(
    num_cores=128,
    neurons_per_core=1024,
    synapses_per_core=1024*1024,
    timestep=1.0,  # ms
    voltage_precision=23,  # bit
    current_precision=23   # bit
)

# Configura chip neuromorfico
neuromorphic_chip = NeuromorphicChip(
    architecture="loihi",
    num_cores=64,
    power_budget=1.0,  # Watt
    memory_capacity=1024*1024,  # synapses
    communication_bandwidth=1000  # MB/s
)

# Mappa rete su hardware neuromorfico
hardware_mapping = await loihi_sim.map_network_to_hardware(
    snn,
    optimization_objective="minimize_power",  # "minimize_power", "minimize_latency", "maximize_throughput"
    constraints={
        "max_cores": 64,
        "max_power": 0.5,  # Watt
        "max_latency": 10.0  # ms
    }
)

print(f"Hardware Mapping Results:")
print(f"  Cores used: {hardware_mapping.cores_used}/{neuromorphic_chip.num_cores}")
print(f"  Power consumption: {hardware_mapping.power_consumption:.3f} W")
print(f"  Estimated latency: {hardware_mapping.latency:.2f} ms")
print(f"  Memory utilization: {hardware_mapping.memory_utilization:.1%}")

# Simula esecuzione su hardware
execution_result = await loihi_sim.execute_network(
    mapped_network=hardware_mapping,
    input_data=input_spikes,
    simulation_time=100.0,  # ms
    power_monitoring=True
)

print(f"\nHardware Execution Results:")
print(f"  Execution time: {execution_result.execution_time:.2f} ms")
print(f"  Energy consumption: {execution_result.energy_consumption:.6f} J")
print(f"  Throughput: {execution_result.throughput:.0f} inferences/s")
print(f"  Accuracy: {execution_result.accuracy:.3f}")

# Confronto con implementazione software
software_result = await snn.execute_software(
    input_spikes,
    simulation_time=100.0
)

print(f"\nSoftware vs Hardware Comparison:")
print(f"  Speed-up: {software_result.execution_time / execution_result.execution_time:.1f}x")
print(f"  Energy efficiency: {software_result.energy / execution_result.energy_consumption:.0f}x")
print(f"  Accuracy difference: {abs(software_result.accuracy - execution_result.accuracy):.4f}")

# Simulatore SpiNNaker
spinnaker_sim = SpiNNakerSimulator(
    num_boards=1,
    chips_per_board=48,
    cores_per_chip=18,
    memory_per_core=64*1024,  # bytes
    router_entries=1024
)

# Mappa rete su SpiNNaker
spinnaker_mapping = await spinnaker_sim.map_network(
    snn,
    partitioning_strategy="population_based",
    routing_algorithm="ner",  # Nearest Neighbour Routing
    placement_algorithm="sa"   # Simulated Annealing
)

print(f"\nSpiNNaker Mapping:")
print(f"  Chips used: {spinnaker_mapping.chips_used}")
print(f"  Cores used: {spinnaker_mapping.cores_used}")
print(f"  Router entries: {spinnaker_mapping.router_entries_used}")
print(f"  Communication overhead: {spinnaker_mapping.comm_overhead:.1%}")
```

## Bio-Inspired AI

### Evolutionary Algorithms

```python
from src.emerging.bio_inspired_ai.evolutionary_algorithms import (
    GeneticAlgorithm, DifferentialEvolution, ParticleSwarmOptimization, 
    EvolutionStrategy, NSGA2
)
import numpy as np

# Algoritmo Genetico
ga = GeneticAlgorithm(
    population_size=100,
    chromosome_length=50,
    crossover_rate=0.8,
    mutation_rate=0.01,
    selection_method="tournament",  # "tournament", "roulette", "rank"
    crossover_method="uniform",     # "uniform", "single_point", "two_point"
    mutation_method="bit_flip",     # "bit_flip", "gaussian", "polynomial"
    elitism=True
)

# Funzione obiettivo (esempio: ottimizzazione di Rastrigin)
def rastrigin_function(x):
    """Funzione di Rastrigin - problema di ottimizzazione multimodale"""
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Inizializza popolazione
population = await ga.initialize_population(
    bounds=[(-5.12, 5.12)] * 20,  # 20 variabili, range [-5.12, 5.12]
    initialization_method="random"
)

print(f"Population initialized: {len(population)} individuals")
print(f"Chromosome length: {len(population[0])}")

# Evoluzione
evolution_history = []
for generation in range(200):
    # Valuta fitness
    fitness_scores = []
    for individual in population:
        fitness = -rastrigin_function(individual)  # Minimizzazione -> massimizzazione
        fitness_scores.append(fitness)
    
    # Statistiche generazione
    best_fitness = max(fitness_scores)
    avg_fitness = np.mean(fitness_scores)
    evolution_history.append({
        "generation": generation,
        "best_fitness": best_fitness,
        "avg_fitness": avg_fitness,
        "diversity": ga.calculate_diversity(population)
    })
    
    if generation % 20 == 0:
        print(f"Generation {generation}: Best = {best_fitness:.4f}, "
              f"Avg = {avg_fitness:.4f}, Diversity = {evolution_history[-1]['diversity']:.3f}")
    
    # Selezione
    selected_parents = await ga.selection(
        population,
        fitness_scores,
        num_parents=80
    )
    
    # Crossover
    offspring = await ga.crossover(
        selected_parents,
        offspring_size=population_size - len(selected_parents)
    )
    
    # Mutazione
    mutated_offspring = await ga.mutation(offspring)
    
    # Nuova popolazione (elitismo)
    if ga.elitism:
        # Mantieni i migliori individui
        elite_indices = np.argsort(fitness_scores)[-20:]  # Top 20
        elite = [population[i] for i in elite_indices]
        population = elite + mutated_offspring[:80]
    else:
        population = selected_parents + mutated_offspring

# Risultati finali
best_individual = population[np.argmax(fitness_scores)]
best_value = rastrigin_function(best_individual)

print(f"\nGenetic Algorithm Results:")
print(f"  Best solution: {best_individual[:5]}...")
print(f"  Best value: {best_value:.6f}")
print(f"  Global optimum: 0.0")
print(f"  Error: {abs(best_value):.6f}")

# Evoluzione Differenziale
de = DifferentialEvolution(
    population_size=50,
    differential_weight=0.8,
    crossover_probability=0.9,
    strategy="rand/1/bin",  # "rand/1/bin", "best/1/bin", "current-to-best/1/bin"
    bounds=[(-5.12, 5.12)] * 20
)

# Ottimizzazione con DE
de_result = await de.optimize(
    objective_function=rastrigin_function,
    max_iterations=1000,
    tolerance=1e-6
)

print(f"\nDifferential Evolution Results:")
print(f"  Best solution: {de_result.best_solution[:5]}...")
print(f"  Best value: {de_result.best_value:.6f}")
print(f"  Convergence: {de_result.converged}")
print(f"  Iterations: {de_result.iterations}")

# Particle Swarm Optimization
pso = ParticleSwarmOptimization(
    num_particles=30,
    inertia_weight=0.9,
    cognitive_coefficient=2.0,
    social_coefficient=2.0,
    bounds=[(-5.12, 5.12)] * 20,
    velocity_clamp=0.5
)

# Ottimizzazione con PSO
pso_result = await pso.optimize(
    objective_function=rastrigin_function,
    max_iterations=500,
    tolerance=1e-6
)

print(f"\nParticle Swarm Optimization Results:")
print(f"  Best solution: {pso_result.best_solution[:5]}...")
print(f"  Best value: {pso_result.best_value:.6f}")
print(f"  Convergence: {pso_result.converged}")
print(f"  Swarm diversity: {pso_result.final_diversity:.3f}")

# Ottimizzazione multi-obiettivo con NSGA-II
nsga2 = NSGA2(
    population_size=100,
    num_objectives=2,
    crossover_rate=0.9,
    mutation_rate=0.1,
    bounds=[(-5, 5)] * 10
)

# Funzioni obiettivo multi-obiettivo (esempio: ZDT1)
def zdt1_objectives(x):
    """Funzioni obiettivo ZDT1"""
    f1 = x[0]
    g = 1 + 9 * sum(x[1:]) / (len(x) - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return [f1, f2]

# Ottimizzazione multi-obiettivo
pareto_front = await nsga2.optimize(
    objective_functions=zdt1_objectives,
    max_generations=250
)

print(f"\nNSGA-II Multi-Objective Optimization:")
print(f"  Pareto front size: {len(pareto_front.solutions)}")
print(f"  Hypervolume: {pareto_front.hypervolume:.4f}")
print(f"  Spread metric: {pareto_front.spread:.4f}")
print(f"  Convergence metric: {pareto_front.convergence:.4f}")
```

### Swarm Intelligence

```python
from src.emerging.bio_inspired_ai.swarm_intelligence import (
    AntColonyOptimization, BeeAlgorithm, FireflyAlgorithm, CuckooSearch
)

# Ant Colony Optimization per TSP
aco = AntColonyOptimization(
    num_ants=50,
    alpha=1.0,      # Importanza feromone
    beta=2.0,       # Importanza euristica
    rho=0.1,        # Tasso evaporazione feromone
    q0=0.9,         # Parametro exploitation vs exploration
    tau0=0.1        # Feromone iniziale
)

# Problema TSP (Traveling Salesman Problem)
cities = np.random.rand(20, 2) * 100  # 20 città casuali
distance_matrix = np.zeros((20, 20))

for i in range(20):
    for j in range(20):
        if i != j:
            distance_matrix[i][j] = np.linalg.norm(cities[i] - cities[j])

print(f"TSP problem: {len(cities)} cities")
print(f"Distance matrix shape: {distance_matrix.shape}")

# Risolvi TSP con ACO
tsp_solution = await aco.solve_tsp(
    distance_matrix,
    max_iterations=200,
    convergence_threshold=50  # Iterazioni senza miglioramento
)

print(f"\nAnt Colony Optimization (TSP):")
print(f"  Best tour length: {tsp_solution.best_distance:.2f}")
print(f"  Best tour: {tsp_solution.best_tour[:10]}...")
print(f"  Convergence iteration: {tsp_solution.convergence_iteration}")
print(f"  Final pheromone diversity: {tsp_solution.pheromone_diversity:.3f}")

# Bee Algorithm per ottimizzazione continua
bee_algorithm = BeeAlgorithm(
    num_scout_bees=10,
    num_selected_sites=5,
    num_elite_sites=2,
    num_bees_around_elite=10,
    num_bees_around_selected=5,
    initial_patch_size=1.0,
    shrinking_factor=0.95
)

# Ottimizzazione funzione Ackley
def ackley_function(x):
    """Funzione di Ackley"""
    a, b, c = 20, 0.2, 2*np.pi
    n = len(x)
    sum1 = sum([xi**2 for xi in x])
    sum2 = sum([np.cos(c*xi) for xi in x])
    return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.exp(1)

bee_result = await bee_algorithm.optimize(
    objective_function=ackley_function,
    bounds=[(-32.768, 32.768)] * 10,
    max_iterations=300
)

print(f"\nBee Algorithm Results:")
print(f"  Best solution: {bee_result.best_solution[:5]}...")
print(f"  Best value: {bee_result.best_value:.6f}")
print(f"  Global optimum: 0.0")
print(f"  Iterations: {bee_result.iterations}")

# Firefly Algorithm
firefly_algorithm = FireflyAlgorithm(
    num_fireflies=25,
    alpha=0.2,      # Randomness
    beta0=1.0,      # Attractiveness
    gamma=1.0,      # Light absorption
    bounds=[(-5, 5)] * 20
)

# Ottimizzazione funzione Sphere
def sphere_function(x):
    """Funzione Sphere"""
    return sum([xi**2 for xi in x])

firefly_result = await firefly_algorithm.optimize(
    objective_function=sphere_function,
    max_iterations=200
)

print(f"\nFirefly Algorithm Results:")
print(f"  Best solution: {firefly_result.best_solution[:5]}...")
print(f"  Best value: {firefly_result.best_value:.6f}")
print(f"  Light intensity distribution: {firefly_result.light_distribution}")

# Cuckoo Search
cuckoo_search = CuckooSearch(
    num_nests=25,
    discovery_rate=0.25,  # Probabilità scoperta uova estranee
    levy_flight_parameter=1.5,
    bounds=[(-10, 10)] * 15
)

# Ottimizzazione funzione Rosenbrock
def rosenbrock_function(x):
    """Funzione di Rosenbrock"""
    return sum([100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1)])

cuckoo_result = await cuckoo_search.optimize(
    objective_function=rosenbrock_function,
    max_iterations=500
)

print(f"\nCuckoo Search Results:")
print(f"  Best solution: {cuckoo_result.best_solution[:5]}...")
print(f"  Best value: {cuckoo_result.best_value:.6f}")
print(f"  Global optimum: 0.0")
print(f"  Levy flights performed: {cuckoo_result.levy_flights_count}")
```