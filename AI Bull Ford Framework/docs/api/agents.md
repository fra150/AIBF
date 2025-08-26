# Agents API Reference

## Panoramica

Il modulo Agents di AIBF fornisce un framework completo per la creazione di agenti intelligenti autonomi, inclusi sistemi di planning, coordinamento multi-agente e capacità di autonomia avanzate.

## Planning

### Goal-Oriented Action Planning (GOAP)

```python
from src.agents.planning.goap import GOAPPlanner, GOAPAction, GOAPGoal, WorldState

# Definisci azioni disponibili
class MoveAction(GOAPAction):
    def __init__(self):
        super().__init__(
            name="move",
            cost=1.0,
            preconditions={"at_location": False},
            effects={"at_location": True}
        )
    
    async def execute(self, agent, world_state):
        # Implementa logica di movimento
        new_state = world_state.copy()
        new_state.set("at_location", True)
        new_state.set("position", self.target_position)
        return new_state
    
    def is_applicable(self, world_state):
        return not world_state.get("at_location", False)

class PickupAction(GOAPAction):
    def __init__(self):
        super().__init__(
            name="pickup",
            cost=2.0,
            preconditions={"at_location": True, "has_item": False},
            effects={"has_item": True}
        )
    
    async def execute(self, agent, world_state):
        new_state = world_state.copy()
        new_state.set("has_item", True)
        new_state.set("inventory_count", world_state.get("inventory_count", 0) + 1)
        return new_state

# Configura planner
planner = GOAPPlanner(
    max_depth=10,
    heuristic_weight=1.0,
    timeout=30.0
)

# Registra azioni
planner.register_action(MoveAction())
planner.register_action(PickupAction())

# Definisci stato iniziale e obiettivo
initial_state = WorldState({
    "at_location": False,
    "has_item": False,
    "position": (0, 0),
    "inventory_count": 0
})

goal = GOAPGoal({
    "has_item": True,
    "inventory_count": 1
})

# Genera piano
plan = await planner.plan(
    initial_state=initial_state,
    goal=goal,
    available_actions=planner.actions
)

if plan.success:
    print(f"Piano trovato con costo {plan.total_cost}:")
    for i, action in enumerate(plan.actions):
        print(f"  {i+1}. {action.name}")
    
    # Esegui piano
    current_state = initial_state
    for action in plan.actions:
        current_state = await action.execute(None, current_state)
        print(f"Eseguita azione: {action.name}")
else:
    print("Nessun piano trovato")
```

### Hierarchical Task Network (HTN)

```python
from src.agents.planning.htn import HTNPlanner, HTNTask, HTNMethod, HTNOperator

# Definisci operatori primitivi
class MoveOperator(HTNOperator):
    def __init__(self, target):
        super().__init__(
            name=f"move_to_{target}",
            preconditions={"location": lambda x: x != target},
            effects={"location": target}
        )
        self.target = target
    
    async def execute(self, agent, state):
        # Implementa movimento
        new_state = state.copy()
        new_state["location"] = self.target
        return new_state

class PickupOperator(HTNOperator):
    def __init__(self, item):
        super().__init__(
            name=f"pickup_{item}",
            preconditions={
                "location": lambda x: x == f"{item}_location",
                "inventory": lambda x: item not in x
            },
            effects={"inventory": lambda x: x + [item]}
        )
        self.item = item

# Definisci metodi per task complessi
class GetItemMethod(HTNMethod):
    def __init__(self, item):
        super().__init__(
            name=f"get_{item}_method",
            task_name=f"get_{item}",
            preconditions={},
            subtasks=[
                HTNTask(f"move_to_{item}_location"),
                HTNTask(f"pickup_{item}")
            ]
        )

# Configura planner HTN
htn_planner = HTNPlanner()

# Registra operatori e metodi
htn_planner.register_operator(MoveOperator("kitchen"))
htn_planner.register_operator(MoveOperator("bedroom"))
htn_planner.register_operator(PickupOperator("key"))
htn_planner.register_method(GetItemMethod("key"))

# Definisci task principale
main_task = HTNTask("get_key")

# Genera piano
initial_state = {
    "location": "living_room",
    "inventory": [],
    "key_location": "kitchen"
}

plan = await htn_planner.plan(
    tasks=[main_task],
    initial_state=initial_state
)

if plan.success:
    print("Piano HTN generato:")
    for step in plan.steps:
        print(f"  - {step.operator.name}")
else:
    print("Pianificazione HTN fallita")
```

### Monte Carlo Tree Search (MCTS)

```python
from src.agents.planning.mcts import MCTSPlanner, MCTSNode, MCTSConfig

# Configura MCTS
mcts_config = MCTSConfig(
    exploration_constant=1.414,  # C parameter
    max_iterations=1000,
    max_depth=50,
    simulation_policy="random",  # "random", "heuristic", "neural"
    backup_strategy="average",   # "average", "max", "robust"
    early_termination=True
)

# Definisci ambiente per MCTS
class GameEnvironment:
    def __init__(self):
        self.state = self.get_initial_state()
    
    def get_initial_state(self):
        return {"position": (0, 0), "score": 0, "moves": 0}
    
    def get_legal_actions(self, state):
        return ["up", "down", "left", "right"]
    
    def apply_action(self, state, action):
        new_state = state.copy()
        x, y = state["position"]
        
        if action == "up": y += 1
        elif action == "down": y -= 1
        elif action == "left": x -= 1
        elif action == "right": x += 1
        
        new_state["position"] = (x, y)
        new_state["moves"] += 1
        new_state["score"] = self.calculate_score(new_state)
        
        return new_state
    
    def is_terminal(self, state):
        return state["moves"] >= 10 or state["score"] >= 100
    
    def get_reward(self, state):
        if self.is_terminal(state):
            return state["score"]
        return 0
    
    def calculate_score(self, state):
        x, y = state["position"]
        return max(0, 100 - abs(x - 5) - abs(y - 5))  # Obiettivo: raggiungere (5,5)

# Crea planner MCTS
env = GameEnvironment()
mcts_planner = MCTSPlanner(env, mcts_config)

# Esegui ricerca
initial_state = env.get_initial_state()
best_action = await mcts_planner.search(initial_state)

print(f"Migliore azione trovata: {best_action}")

# Ottieni statistiche
stats = mcts_planner.get_search_statistics()
print(f"Nodi esplorati: {stats['nodes_explored']}")
print(f"Profondità media: {stats['average_depth']:.2f}")
print(f"Valore stimato: {stats['estimated_value']:.3f}")
```

### A* Planning

```python
from src.agents.planning.astar import AStarPlanner, AStarNode, Heuristic

# Definisci euristica personalizzata
class ManhattanHeuristic(Heuristic):
    def __init__(self, goal_position):
        self.goal_position = goal_position
    
    def estimate(self, state):
        current_pos = state.get("position", (0, 0))
        goal_pos = self.goal_position
        return abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])

# Definisci spazio degli stati
class GridWorld:
    def __init__(self, width, height, obstacles=None):
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()
    
    def get_neighbors(self, state):
        x, y = state["position"]
        neighbors = []
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = x + dx, y + dy
            
            if (0 <= new_x < self.width and 
                0 <= new_y < self.height and 
                (new_x, new_y) not in self.obstacles):
                
                new_state = state.copy()
                new_state["position"] = (new_x, new_y)
                neighbors.append((new_state, 1.0))  # (state, cost)
        
        return neighbors
    
    def is_goal(self, state, goal_state):
        return state["position"] == goal_state["position"]

# Configura A*
grid = GridWorld(10, 10, obstacles={(3, 3), (3, 4), (4, 3)})
heuristic = ManhattanHeuristic((8, 8))

astar_planner = AStarPlanner(
    state_space=grid,
    heuristic=heuristic,
    max_iterations=1000
)

# Trova percorso
start_state = {"position": (0, 0)}
goal_state = {"position": (8, 8)}

path = await astar_planner.find_path(start_state, goal_state)

if path.found:
    print(f"Percorso trovato con costo {path.total_cost}:")
    for i, state in enumerate(path.states):
        pos = state["position"]
        print(f"  {i}: {pos}")
else:
    print("Nessun percorso trovato")
```

## Multi-Agent Systems

### Agent Communication

```python
from src.agents.multi_agent.communication import (
    MessageBus, Message, CommunicationProtocol, AgentAddress
)
from src.agents.multi_agent.ontology import Ontology, Concept, Relation

# Definisci ontologia per comunicazione
ontology = Ontology()
ontology.add_concept(Concept("Task", properties=["id", "type", "priority"]))
ontology.add_concept(Concept("Resource", properties=["id", "type", "availability"]))
ontology.add_relation(Relation("requires", "Task", "Resource"))

# Configura bus di messaggi
message_bus = MessageBus(
    ontology=ontology,
    max_message_size=1024*1024,  # 1MB
    message_ttl=300,  # 5 minuti
    delivery_guarantee="at_least_once"  # "at_most_once", "exactly_once"
)

# Definisci protocollo di comunicazione
class TaskAllocationProtocol(CommunicationProtocol):
    def __init__(self):
        super().__init__("task_allocation")
    
    async def handle_task_request(self, sender, message):
        task_data = message.content["task"]
        
        # Logica di allocazione task
        if self.can_handle_task(task_data):
            response = Message(
                sender=self.agent_address,
                receiver=sender,
                protocol="task_allocation",
                performative="accept",
                content={"task_id": task_data["id"], "estimated_time": 30}
            )
        else:
            response = Message(
                sender=self.agent_address,
                receiver=sender,
                protocol="task_allocation",
                performative="refuse",
                content={"task_id": task_data["id"], "reason": "insufficient_resources"}
            )
        
        await self.send_message(response)
    
    def can_handle_task(self, task_data):
        # Implementa logica di verifica capacità
        return task_data["type"] in self.supported_task_types

# Registra protocollo
message_bus.register_protocol(TaskAllocationProtocol())

# Invia messaggio
sender_address = AgentAddress("agent_1", "localhost", 8001)
receiver_address = AgentAddress("agent_2", "localhost", 8002)

message = Message(
    sender=sender_address,
    receiver=receiver_address,
    protocol="task_allocation",
    performative="request",
    content={
        "task": {
            "id": "task_123",
            "type": "data_processing",
            "priority": 0.8,
            "deadline": "2024-01-15T10:00:00Z"
        }
    }
)

await message_bus.send_message(message)
```

### Coordination Mechanisms

#### Contract Net Protocol

```python
from src.agents.multi_agent.coordination.contract_net import (
    ContractNetManager, ContractNetInitiator, ContractNetParticipant
)

# Definisci task per contract net
class DataProcessingTask:
    def __init__(self, task_id, data_size, deadline):
        self.task_id = task_id
        self.data_size = data_size
        self.deadline = deadline
        self.requirements = {
            "min_memory": data_size * 2,
            "min_cpu_cores": 4,
            "required_software": ["python", "pandas"]
        }

# Agente iniziatore
class TaskManagerAgent(ContractNetInitiator):
    def __init__(self, agent_id):
        super().__init__(agent_id)
        self.pending_tasks = []
    
    async def create_call_for_proposals(self, task):
        return {
            "task_id": task.task_id,
            "description": f"Process {task.data_size}MB of data",
            "requirements": task.requirements,
            "deadline": task.deadline,
            "max_price": 100.0
        }
    
    async def evaluate_proposals(self, proposals):
        # Valuta proposte basandosi su prezzo, tempo e affidabilità
        scored_proposals = []
        
        for proposal in proposals:
            score = self.calculate_proposal_score(proposal)
            scored_proposals.append((proposal, score))
        
        # Ordina per punteggio decrescente
        scored_proposals.sort(key=lambda x: x[1], reverse=True)
        
        return scored_proposals[0][0] if scored_proposals else None
    
    def calculate_proposal_score(self, proposal):
        price_score = max(0, 1 - proposal["price"] / 100.0)
        time_score = max(0, 1 - proposal["estimated_time"] / 3600)
        reliability_score = proposal.get("reliability", 0.5)
        
        return 0.4 * price_score + 0.4 * time_score + 0.2 * reliability_score

# Agente partecipante
class WorkerAgent(ContractNetParticipant):
    def __init__(self, agent_id, capabilities):
        super().__init__(agent_id)
        self.capabilities = capabilities
        self.current_load = 0.0
    
    async def evaluate_cfp(self, cfp):
        # Valuta se può gestire il task
        if not self.meets_requirements(cfp["requirements"]):
            return None
        
        if self.current_load > 0.8:
            return None
        
        # Calcola proposta
        estimated_time = self.estimate_processing_time(cfp)
        price = self.calculate_price(cfp, estimated_time)
        
        return {
            "agent_id": self.agent_id,
            "price": price,
            "estimated_time": estimated_time,
            "reliability": self.get_reliability_score(),
            "start_time": "immediate"
        }
    
    def meets_requirements(self, requirements):
        return (
            self.capabilities["memory"] >= requirements["min_memory"] and
            self.capabilities["cpu_cores"] >= requirements["min_cpu_cores"] and
            all(sw in self.capabilities["software"] for sw in requirements["required_software"])
        )
    
    async def execute_task(self, task_details):
        # Implementa esecuzione del task
        self.current_load += 0.3
        
        try:
            # Simula elaborazione
            await asyncio.sleep(task_details["estimated_time"] / 100)
            result = {"status": "completed", "output_size": "50MB"}
            return result
        finally:
            self.current_load -= 0.3

# Configura contract net
contract_manager = ContractNetManager()

# Crea agenti
task_manager = TaskManagerAgent("manager_1")
worker1 = WorkerAgent("worker_1", {
    "memory": 16000,
    "cpu_cores": 8,
    "software": ["python", "pandas", "numpy"]
})
worker2 = WorkerAgent("worker_2", {
    "memory": 8000,
    "cpu_cores": 4,
    "software": ["python", "pandas"]
})

# Registra agenti
contract_manager.register_initiator(task_manager)
contract_manager.register_participant(worker1)
contract_manager.register_participant(worker2)

# Esegui contract net
task = DataProcessingTask("task_001", 5000, "2024-01-15T12:00:00Z")
result = await contract_manager.execute_contract_net(
    initiator=task_manager,
    task=task,
    timeout=60.0
)

if result.success:
    print(f"Task assegnato a {result.winner.agent_id}")
    print(f"Prezzo: {result.winning_proposal['price']}")
    print(f"Tempo stimato: {result.winning_proposal['estimated_time']}s")
else:
    print(f"Contract net fallito: {result.failure_reason}")
```

#### Auction Mechanisms

```python
from src.agents.multi_agent.coordination.auctions import (
    EnglishAuction, DutchAuction, SealedBidAuction, VickreyAuction
)

# Asta inglese (ascending price)
class ResourceAuction(EnglishAuction):
    def __init__(self, resource_id, starting_price, increment):
        super().__init__(
            auction_id=f"auction_{resource_id}",
            starting_price=starting_price,
            bid_increment=increment,
            timeout=300  # 5 minuti
        )
        self.resource_id = resource_id
    
    def validate_bid(self, bidder, amount):
        # Valida che l'offerente abbia fondi sufficienti
        return bidder.get_available_funds() >= amount

# Agente offerente
class BiddingAgent:
    def __init__(self, agent_id, budget, valuation_function):
        self.agent_id = agent_id
        self.budget = budget
        self.valuation_function = valuation_function
        self.active_bids = {}
    
    def get_available_funds(self):
        committed_funds = sum(self.active_bids.values())
        return self.budget - committed_funds
    
    async def decide_bid(self, auction, current_price):
        # Strategia di bidding basata su valutazione
        resource_value = self.valuation_function(auction.resource_id)
        
        if current_price >= resource_value:
            return None  # Non fare offerta
        
        # Strategia conservativa: offri fino al 90% del valore
        max_bid = resource_value * 0.9
        next_bid = current_price + auction.bid_increment
        
        if next_bid <= max_bid and next_bid <= self.get_available_funds():
            return next_bid
        
        return None
    
    async def on_auction_won(self, auction, winning_bid):
        print(f"Agente {self.agent_id} ha vinto l'asta {auction.auction_id} con offerta {winning_bid}")
        self.budget -= winning_bid
    
    async def on_auction_lost(self, auction, winning_bid):
        print(f"Agente {self.agent_id} ha perso l'asta {auction.auction_id}")

# Configura asta
auction = ResourceAuction(
    resource_id="gpu_cluster_1",
    starting_price=50.0,
    increment=5.0
)

# Crea agenti offerenti
agent1 = BiddingAgent("bidder_1", 200.0, lambda r: 150.0)  # Valuta risorsa a 150
agent2 = BiddingAgent("bidder_2", 180.0, lambda r: 120.0)  # Valuta risorsa a 120
agent3 = BiddingAgent("bidder_3", 250.0, lambda r: 180.0)  # Valuta risorsa a 180

# Registra offerenti
auction.register_bidder(agent1)
auction.register_bidder(agent2)
auction.register_bidder(agent3)

# Esegui asta
result = await auction.run()

if result.winner:
    print(f"Asta completata. Vincitore: {result.winner.agent_id}")
    print(f"Prezzo finale: {result.winning_bid}")
    print(f"Numero di round: {result.num_rounds}")
else:
    print("Asta fallita - nessuna offerta valida")
```

### Consensus Algorithms

```python
from src.agents.multi_agent.consensus import (
    RaftConsensus, PBFTConsensus, ConsensusNode, ConsensusMessage
)

# Implementazione Raft
class RaftNode(ConsensusNode):
    def __init__(self, node_id, cluster_nodes):
        super().__init__(node_id)
        self.cluster_nodes = cluster_nodes
        self.state = "follower"  # "follower", "candidate", "leader"
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
    
    async def start_election(self):
        self.state = "candidate"
        self.current_term += 1
        self.voted_for = self.node_id
        
        votes = 1  # Voto per se stesso
        
        # Richiedi voti da altri nodi
        for node in self.cluster_nodes:
            if node != self.node_id:
                vote_request = ConsensusMessage(
                    type="vote_request",
                    term=self.current_term,
                    candidate_id=self.node_id,
                    last_log_index=len(self.log) - 1,
                    last_log_term=self.log[-1]["term"] if self.log else 0
                )
                
                response = await self.send_message(node, vote_request)
                if response and response.vote_granted:
                    votes += 1
        
        # Diventa leader se ha la maggioranza
        if votes > len(self.cluster_nodes) // 2:
            self.state = "leader"
            await self.send_heartbeats()
    
    async def handle_vote_request(self, message):
        vote_granted = (
            message.term > self.current_term and
            (self.voted_for is None or self.voted_for == message.candidate_id) and
            self.is_log_up_to_date(message.last_log_index, message.last_log_term)
        )
        
        if vote_granted:
            self.voted_for = message.candidate_id
            self.current_term = message.term
        
        return ConsensusMessage(
            type="vote_response",
            term=self.current_term,
            vote_granted=vote_granted
        )
    
    async def append_entry(self, entry):
        if self.state == "leader":
            # Aggiungi entry al log locale
            log_entry = {
                "term": self.current_term,
                "data": entry,
                "index": len(self.log)
            }
            self.log.append(log_entry)
            
            # Replica su altri nodi
            success_count = 1  # Leader conta come successo
            
            for node in self.cluster_nodes:
                if node != self.node_id:
                    append_request = ConsensusMessage(
                        type="append_entries",
                        term=self.current_term,
                        leader_id=self.node_id,
                        prev_log_index=len(self.log) - 2,
                        prev_log_term=self.log[-2]["term"] if len(self.log) > 1 else 0,
                        entries=[log_entry],
                        leader_commit=self.commit_index
                    )
                    
                    response = await self.send_message(node, append_request)
                    if response and response.success:
                        success_count += 1
            
            # Commit se replicato sulla maggioranza
            if success_count > len(self.cluster_nodes) // 2:
                self.commit_index = len(self.log) - 1
                return True
        
        return False

# Configura cluster Raft
cluster_nodes = ["node_1", "node_2", "node_3", "node_4", "node_5"]
raft_nodes = {}

for node_id in cluster_nodes:
    raft_nodes[node_id] = RaftNode(node_id, cluster_nodes)

# Simula consenso
leader_node = raft_nodes["node_1"]
await leader_node.start_election()

if leader_node.state == "leader":
    # Aggiungi entry al log distribuito
    success = await leader_node.append_entry({
        "operation": "set",
        "key": "config_value",
        "value": "new_setting"
    })
    
    if success:
        print("Entry replicata con successo nel cluster")
    else:
        print("Fallimento nella replicazione")
```

## Autonomy

### Self-Monitoring

```python
from src.agents.autonomy.self_monitoring import (
    SelfMonitor, PerformanceMetric, HealthCheck, AnomalyDetector
)

# Definisci metriche di performance
class ResponseTimeMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("response_time", "seconds")
        self.response_times = []
    
    def record_measurement(self, value):
        self.response_times.append(value)
        if len(self.response_times) > 1000:
            self.response_times.pop(0)  # Mantieni solo ultimi 1000
    
    def get_current_value(self):
        return np.mean(self.response_times[-10:]) if self.response_times else 0
    
    def get_trend(self):
        if len(self.response_times) < 20:
            return "stable"
        
        recent = np.mean(self.response_times[-10:])
        older = np.mean(self.response_times[-20:-10])
        
        if recent > older * 1.2:
            return "degrading"
        elif recent < older * 0.8:
            return "improving"
        else:
            return "stable"

class MemoryUsageMetric(PerformanceMetric):
    def __init__(self):
        super().__init__("memory_usage", "MB")
    
    def get_current_value(self):
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

# Health checks
class DatabaseHealthCheck(HealthCheck):
    def __init__(self, db_connection):
        super().__init__("database_connectivity")
        self.db_connection = db_connection
    
    async def check_health(self):
        try:
            await self.db_connection.execute("SELECT 1")
            return True, "Database connection OK"
        except Exception as e:
            return False, f"Database error: {str(e)}"

class APIHealthCheck(HealthCheck):
    def __init__(self, api_endpoint):
        super().__init__("api_availability")
        self.api_endpoint = api_endpoint
    
    async def check_health(self):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_endpoint}/health", timeout=5) as response:
                    if response.status == 200:
                        return True, "API endpoint healthy"
                    else:
                        return False, f"API returned status {response.status}"
        except Exception as e:
            return False, f"API unreachable: {str(e)}"

# Rilevamento anomalie
class StatisticalAnomalyDetector(AnomalyDetector):
    def __init__(self, metric_name, window_size=100, threshold=2.0):
        super().__init__(metric_name)
        self.window_size = window_size
        self.threshold = threshold
        self.values = []
    
    def detect_anomaly(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        
        if len(self.values) < 10:
            return False, "Insufficient data"
        
        mean = np.mean(self.values[:-1])  # Escludi valore corrente
        std = np.std(self.values[:-1])
        
        if std == 0:
            return False, "No variation in data"
        
        z_score = abs(value - mean) / std
        
        if z_score > self.threshold:
            return True, f"Anomaly detected: z-score = {z_score:.2f}"
        
        return False, "Normal value"

# Configura self-monitoring
self_monitor = SelfMonitor(
    agent_id="autonomous_agent_1",
    monitoring_interval=30.0,  # 30 secondi
    alert_threshold=0.8
)

# Registra metriche
self_monitor.register_metric(ResponseTimeMetric())
self_monitor.register_metric(MemoryUsageMetric())

# Registra health checks
self_monitor.register_health_check(DatabaseHealthCheck(db_connection))
self_monitor.register_health_check(APIHealthCheck("http://api.example.com"))

# Registra rilevatori di anomalie
self_monitor.register_anomaly_detector(
    StatisticalAnomalyDetector("response_time", threshold=2.5)
)
self_monitor.register_anomaly_detector(
    StatisticalAnomalyDetector("memory_usage", threshold=3.0)
)

# Avvia monitoring
await self_monitor.start_monitoring()

# Simula operazioni dell'agente
for i in range(100):
    start_time = time.time()
    
    # Simula operazione
    await asyncio.sleep(random.uniform(0.1, 0.5))
    
    # Registra tempo di risposta
    response_time = time.time() - start_time
    self_monitor.record_metric("response_time", response_time)
    
    await asyncio.sleep(1)

# Ottieni report di monitoring
report = await self_monitor.get_monitoring_report()
print(f"Agent Health: {report.overall_health}")
print(f"Active Alerts: {len(report.active_alerts)}")
print(f"Performance Trends: {report.performance_trends}")
```

### Self-Adaptation

```python
from src.agents.autonomy.self_adaptation import (
    AdaptationEngine, AdaptationRule, AdaptationAction, AdaptationTrigger
)

# Definisci trigger per adattamento
class HighLatencyTrigger(AdaptationTrigger):
    def __init__(self, threshold=1.0):
        super().__init__("high_latency")
        self.threshold = threshold
    
    def should_trigger(self, context):
        current_latency = context.get_metric("response_time")
        return current_latency > self.threshold

class LowMemoryTrigger(AdaptationTrigger):
    def __init__(self, threshold=0.9):
        super().__init__("low_memory")
        self.threshold = threshold  # 90% di utilizzo
    
    def should_trigger(self, context):
        memory_usage = context.get_metric("memory_usage_percent")
        return memory_usage > self.threshold

# Definisci azioni di adattamento
class ReduceBatchSizeAction(AdaptationAction):
    def __init__(self):
        super().__init__("reduce_batch_size")
    
    async def execute(self, context):
        current_batch_size = context.get_parameter("batch_size")
        new_batch_size = max(1, current_batch_size // 2)
        
        context.set_parameter("batch_size", new_batch_size)
        
        return {
            "action": "reduce_batch_size",
            "old_value": current_batch_size,
            "new_value": new_batch_size,
            "timestamp": time.time()
        }

class IncreaseTimeoutAction(AdaptationAction):
    def __init__(self):
        super().__init__("increase_timeout")
    
    async def execute(self, context):
        current_timeout = context.get_parameter("request_timeout")
        new_timeout = min(60.0, current_timeout * 1.5)
        
        context.set_parameter("request_timeout", new_timeout)
        
        return {
            "action": "increase_timeout",
            "old_value": current_timeout,
            "new_value": new_timeout,
            "timestamp": time.time()
        }

class GarbageCollectionAction(AdaptationAction):
    def __init__(self):
        super().__init__("force_gc")
    
    async def execute(self, context):
        import gc
        collected = gc.collect()
        
        return {
            "action": "garbage_collection",
            "objects_collected": collected,
            "timestamp": time.time()
        }

# Definisci regole di adattamento
latency_rule = AdaptationRule(
    name="high_latency_response",
    trigger=HighLatencyTrigger(threshold=2.0),
    actions=[
        ReduceBatchSizeAction(),
        IncreaseTimeoutAction()
    ],
    cooldown_period=60.0,  # Non riattivare per 60 secondi
    priority=0.8
)

memory_rule = AdaptationRule(
    name="memory_pressure_response",
    trigger=LowMemoryTrigger(threshold=0.85),
    actions=[
        GarbageCollectionAction(),
        ReduceBatchSizeAction()
    ],
    cooldown_period=30.0,
    priority=0.9  # Priorità più alta
)

# Configura adaptation engine
adaptation_engine = AdaptationEngine(
    agent_id="adaptive_agent_1",
    evaluation_interval=10.0,  # Valuta ogni 10 secondi
    max_concurrent_adaptations=2
)

# Registra regole
adaptation_engine.register_rule(latency_rule)
adaptation_engine.register_rule(memory_rule)

# Avvia engine di adattamento
await adaptation_engine.start()

# Simula condizioni che richiedono adattamento
context = adaptation_engine.get_context()
context.set_parameter("batch_size", 32)
context.set_parameter("request_timeout", 5.0)

# Simula metriche che cambiano
for i in range(100):
    # Simula aumento graduale della latenza
    latency = 0.5 + (i / 100) * 2.0
    context.update_metric("response_time", latency)
    
    # Simula utilizzo memoria variabile
    memory_usage = 0.7 + 0.3 * np.sin(i / 10)
    context.update_metric("memory_usage_percent", memory_usage)
    
    await asyncio.sleep(1)

# Ottieni cronologia adattamenti
adaptation_history = adaptation_engine.get_adaptation_history()
print(f"Adattamenti eseguiti: {len(adaptation_history)}")

for adaptation in adaptation_history:
    print(f"  {adaptation['timestamp']}: {adaptation['rule_name']} -> {adaptation['actions']}")
```

### Self-Learning

```python
from src.agents.autonomy.self_learning import (
    OnlineLearner, ExperienceBuffer, LearningStrategy, MetaLearner
)

# Buffer per esperienze
class ExperienceReplay(ExperienceBuffer):
    def __init__(self, capacity=10000):
        super().__init__(capacity)
        self.experiences = []
        self.priorities = []
    
    def add_experience(self, state, action, reward, next_state, done, metadata=None):
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "metadata": metadata or {},
            "timestamp": time.time()
        }
        
        if len(self.experiences) >= self.capacity:
            # Rimuovi esperienza più vecchia
            self.experiences.pop(0)
            self.priorities.pop(0)
        
        self.experiences.append(experience)
        self.priorities.append(abs(reward) + 0.1)  # Priorità basata su reward
    
    def sample_batch(self, batch_size=32):
        if len(self.experiences) < batch_size:
            return self.experiences
        
        # Campionamento prioritizzato
        probabilities = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(
            len(self.experiences),
            size=batch_size,
            p=probabilities,
            replace=False
        )
        
        return [self.experiences[i] for i in indices]

# Strategia di apprendimento online
class ContinualLearningStrategy(LearningStrategy):
    def __init__(self, model, learning_rate=0.001):
        super().__init__("continual_learning")
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_function = torch.nn.MSELoss()
        self.learning_rate = learning_rate
    
    async def update_model(self, experiences):
        if len(experiences) < 10:
            return {"status": "insufficient_data"}
        
        # Prepara batch di training
        states = torch.stack([exp["state"] for exp in experiences])
        actions = torch.tensor([exp["action"] for exp in experiences])
        rewards = torch.tensor([exp["reward"] for exp in experiences])
        next_states = torch.stack([exp["next_state"] for exp in experiences])
        dones = torch.tensor([exp["done"] for exp in experiences])
        
        # Calcola target Q-values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + 0.99 * next_q_values * (1 - dones.float())
        
        # Calcola Q-values correnti
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calcola loss e aggiorna
        loss = self.loss_function(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "status": "updated",
            "loss": loss.item(),
            "samples_processed": len(experiences)
        }
    
    def adapt_learning_rate(self, performance_trend):
        if performance_trend == "improving":
            self.learning_rate *= 1.05  # Aumenta leggermente
        elif performance_trend == "degrading":
            self.learning_rate *= 0.95  # Diminuisci
        
        # Aggiorna optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

# Meta-learner per adattamento strategico
class MetaLearningAgent(MetaLearner):
    def __init__(self):
        super().__init__()
        self.strategy_performance = {}
        self.current_strategy = None
        self.strategy_switch_threshold = 0.1
    
    def evaluate_strategy_performance(self, strategy_name, recent_rewards):
        if len(recent_rewards) < 10:
            return 0.0
        
        # Calcola performance come media mobile
        performance = np.mean(recent_rewards[-50:])
        
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        self.strategy_performance[strategy_name].append(performance)
        
        # Mantieni solo ultimi 20 valori
        if len(self.strategy_performance[strategy_name]) > 20:
            self.strategy_performance[strategy_name].pop(0)
        
        return performance
    
    def should_switch_strategy(self, current_strategy, available_strategies):
        if current_strategy not in self.strategy_performance:
            return False, None
        
        current_performance = np.mean(self.strategy_performance[current_strategy][-5:])
        
        best_strategy = current_strategy
        best_performance = current_performance
        
        for strategy in available_strategies:
            if strategy in self.strategy_performance:
                strategy_performance = np.mean(self.strategy_performance[strategy][-5:])
                if strategy_performance > best_performance + self.strategy_switch_threshold:
                    best_strategy = strategy
                    best_performance = strategy_performance
        
        return best_strategy != current_strategy, best_strategy

# Learner online principale
class AutonomousLearner(OnlineLearner):
    def __init__(self, model, environment):
        super().__init__()
        self.model = model
        self.environment = environment
        self.experience_buffer = ExperienceReplay(capacity=50000)
        self.learning_strategy = ContinualLearningStrategy(model)
        self.meta_learner = MetaLearningAgent()
        self.recent_rewards = []
        self.learning_enabled = True
    
    async def interact_and_learn(self, num_steps=1000):
        state = self.environment.reset()
        
        for step in range(num_steps):
            # Scegli azione (epsilon-greedy)
            if random.random() < 0.1:  # Esplorazione
                action = self.environment.action_space.sample()
            else:  # Sfruttamento
                with torch.no_grad():
                    q_values = self.model(state.unsqueeze(0))
                    action = q_values.argmax().item()
            
            # Esegui azione
            next_state, reward, done, info = self.environment.step(action)
            
            # Memorizza esperienza
            self.experience_buffer.add_experience(
                state, action, reward, next_state, done, info
            )
            
            self.recent_rewards.append(reward)
            if len(self.recent_rewards) > 1000:
                self.recent_rewards.pop(0)
            
            # Apprendi da esperienze
            if self.learning_enabled and len(self.experience_buffer.experiences) > 100:
                if step % 10 == 0:  # Apprendi ogni 10 step
                    batch = self.experience_buffer.sample_batch(32)
                    update_result = await self.learning_strategy.update_model(batch)
                    
                    # Valuta performance e adatta strategia
                    if step % 100 == 0:
                        performance = self.meta_learner.evaluate_strategy_performance(
                            "continual_learning", self.recent_rewards
                        )
                        
                        # Adatta learning rate basandosi su performance
                        if len(self.recent_rewards) >= 200:
                            recent_perf = np.mean(self.recent_rewards[-100:])
                            older_perf = np.mean(self.recent_rewards[-200:-100])
                            
                            if recent_perf > older_perf:
                                trend = "improving"
                            elif recent_perf < older_perf * 0.95:
                                trend = "degrading"
                            else:
                                trend = "stable"
                            
                            self.learning_strategy.adapt_learning_rate(trend)
            
            state = next_state
            
            if done:
                state = self.environment.reset()
        
        return {
            "total_steps": num_steps,
            "average_reward": np.mean(self.recent_rewards[-100:]) if self.recent_rewards else 0,
            "learning_updates": len(self.experience_buffer.experiences) // 32
        }

# Uso del sistema di self-learning
model = torch.nn.Sequential(
    torch.nn.Linear(4, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 2)
)

# Simula environment (CartPole)
class SimpleEnvironment:
    def __init__(self):
        self.state = None
        self.action_space = type('ActionSpace', (), {'sample': lambda: random.randint(0, 1)})()
    
    def reset(self):
        self.state = torch.randn(4)
        return self.state
    
    def step(self, action):
        # Simula dinamiche semplici
        next_state = self.state + torch.randn(4) * 0.1
        reward = 1.0 if action == 0 else -0.1  # Preferisci azione 0
        done = random.random() < 0.01  # Episodio termina casualmente
        
        self.state = next_state
        return next_state, reward, done, {}

env = SimpleEnvironment()
learner = AutonomousLearner(model, env)

# Avvia apprendimento autonomo
result = await learner.interact_and_learn(num_steps=5000)
print(f"Apprendimento completato: {result}")
```

## Esempi Completi

### Sistema Multi-Agente per Task Allocation

```python
import asyncio
from src.agents import (
    GOAPPlanner, ContractNetManager, SelfMonitor,
    MessageBus, TaskAllocationProtocol
)

async def setup_multi_agent_system():
    # 1. Configura comunicazione
    message_bus = MessageBus()
    
    # 2. Crea agenti specializzati
    coordinator = CoordinatorAgent("coordinator", message_bus)
    worker1 = WorkerAgent("worker_1", ["data_processing", "analysis"], message_bus)
    worker2 = WorkerAgent("worker_2", ["visualization", "reporting"], message_bus)
    worker3 = WorkerAgent("worker_3", ["data_processing", "machine_learning"], message_bus)
    
    # 3. Configura contract net
    contract_manager = ContractNetManager()
    contract_manager.register_initiator(coordinator)
    contract_manager.register_participant(worker1)
    contract_manager.register_participant(worker2)
    contract_manager.register_participant(worker3)
    
    # 4. Avvia self-monitoring per ogni agente
    for agent in [coordinator, worker1, worker2, worker3]:
        monitor = SelfMonitor(agent.agent_id)
        await monitor.start_monitoring()
        agent.set_monitor(monitor)
    
    # 5. Simula task allocation
    tasks = [
        {"id": "task_1", "type": "data_processing", "priority": 0.8, "data_size": 1000},
        {"id": "task_2", "type": "analysis", "priority": 0.6, "complexity": "medium"},
        {"id": "task_3", "type": "visualization", "priority": 0.9, "chart_type": "dashboard"}
    ]
    
    results = []
    for task in tasks:
        result = await contract_manager.execute_contract_net(
            initiator=coordinator,
            task=task,
            timeout=30.0
        )
        results.append(result)
    
    return results

# Esegui sistema multi-agente
results = await setup_multi_agent_system()
for i, result in enumerate(results):
    if result.success:
        print(f"Task {i+1} assegnato a {result.winner.agent_id}")
    else:
        print(f"Task {i+1} non assegnato: {result.failure_reason}")
```

Per ulteriori dettagli e esempi avanzati, consulta la [documentazione completa](../guides/) e gli [esempi pratici](../examples/).