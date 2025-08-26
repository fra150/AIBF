# Applications API Reference

## Panoramica

Il modulo Applications di AIBF fornisce implementazioni specializzate per diversi domini applicativi: robotica, industria, edge computing, sanità, finanza ed educazione.

## Robotics

### Robot Control

```python
from src.applications.robotics.robot_control import (
    RobotController, JointController, CartesianController, TrajectoryPlanner
)
import numpy as np

# Controller per robot manipolatore
robot_controller = RobotController(
    robot_type="manipulator",
    dof=6,  # Degrees of freedom
    control_frequency=100,  # Hz
    safety_limits={
        "joint_limits": [(-180, 180)] * 6,  # gradi
        "velocity_limits": [90] * 6,         # gradi/sec
        "acceleration_limits": [180] * 6     # gradi/sec²
    }
)

# Inizializza controller
await robot_controller.initialize()

# Controllo delle giunture
joint_controller = JointController(
    num_joints=6,
    control_type="position",  # "position", "velocity", "torque"
    pid_gains={
        "kp": [100, 100, 80, 60, 40, 20],
        "ki": [10, 10, 8, 6, 4, 2],
        "kd": [5, 5, 4, 3, 2, 1]
    }
)

# Muovi a posizione target
target_joints = [0, -90, 90, 0, 90, 0]  # gradi
trajectory = await joint_controller.move_to_position(
    target_joints,
    duration=3.0,
    interpolation="cubic"  # "linear", "cubic", "quintic"
)

print(f"Trajectory generated with {len(trajectory.waypoints)} waypoints")

# Esegui movimento
for waypoint in trajectory.waypoints:
    await robot_controller.execute_joint_command(waypoint)
    await asyncio.sleep(trajectory.dt)

# Controllo cartesiano
cartesian_controller = CartesianController(
    base_frame="base_link",
    end_effector_frame="tool0",
    control_type="pose"  # "pose", "position", "orientation"
)

# Muovi end-effector a posizione cartesiana
target_pose = {
    "position": [0.5, 0.2, 0.3],  # metri
    "orientation": [0, 0, 0, 1]   # quaternione [x, y, z, w]
}

cartesian_trajectory = await cartesian_controller.move_to_pose(
    target_pose,
    duration=2.0,
    path_type="straight_line"  # "straight_line", "circular", "spline"
)

# Esegui movimento cartesiano
for pose in cartesian_trajectory.poses:
    joint_angles = await robot_controller.inverse_kinematics(pose)
    await robot_controller.execute_joint_command(joint_angles)
    await asyncio.sleep(cartesian_trajectory.dt)

# Pianificazione traiettorie complesse
trajectory_planner = TrajectoryPlanner(
    robot_model=robot_controller.robot_model,
    obstacle_avoidance=True,
    optimization_objective="time"  # "time", "energy", "smoothness"
)

# Definisci waypoints
waypoints = [
    {"position": [0.3, 0.0, 0.4], "orientation": [0, 0, 0, 1]},
    {"position": [0.5, 0.2, 0.3], "orientation": [0, 0, 0.707, 0.707]},
    {"position": [0.4, -0.1, 0.5], "orientation": [0, 0, 0, 1]}
]

# Pianifica traiettoria ottimale
optimal_trajectory = await trajectory_planner.plan_trajectory(
    waypoints,
    constraints={
        "max_velocity": 0.5,     # m/s
        "max_acceleration": 2.0, # m/s²
        "max_jerk": 10.0        # m/s³
    }
)

print(f"Optimal trajectory duration: {optimal_trajectory.total_duration:.2f}s")
print(f"Energy cost: {optimal_trajectory.energy_cost:.3f}")
```

### Navigation

```python
from src.applications.robotics.navigation import (
    PathPlanner, LocalPlanner, GlobalPlanner, ObstacleAvoidance, SLAM
)

# Sistema SLAM per mappatura e localizzazione
slam_system = SLAM(
    algorithm="gmapping",  # "gmapping", "hector", "cartographer"
    sensor_config={
        "lidar": {
            "range_max": 30.0,
            "angle_min": -3.14159,
            "angle_max": 3.14159,
            "angle_increment": 0.0174533
        },
        "odometry": {
            "noise_model": "gaussian",
            "position_noise": 0.01,
            "orientation_noise": 0.02
        }
    }
)

# Inizializza SLAM
await slam_system.initialize()

# Pianificatore globale
global_planner = GlobalPlanner(
    algorithm="a_star",  # "a_star", "dijkstra", "rrt", "rrt_star"
    map_resolution=0.05,  # metri per pixel
    heuristic="euclidean"  # "euclidean", "manhattan", "diagonal"
)

# Carica mappa
map_data = await slam_system.get_current_map()
await global_planner.load_map(map_data)

# Pianifica percorso globale
start_pose = {"x": 0.0, "y": 0.0, "theta": 0.0}
goal_pose = {"x": 10.0, "y": 5.0, "theta": 1.57}

global_path = await global_planner.plan_path(start_pose, goal_pose)

if global_path.success:
    print(f"Global path found with {len(global_path.waypoints)} waypoints")
    print(f"Path length: {global_path.total_distance:.2f}m")
else:
    print(f"Global planning failed: {global_path.failure_reason}")

# Pianificatore locale per evitamento ostacoli
local_planner = LocalPlanner(
    algorithm="dwa",  # "dwa", "teb", "mpc"
    robot_config={
        "max_vel_x": 1.0,
        "max_vel_theta": 1.0,
        "acc_lim_x": 2.0,
        "acc_lim_theta": 2.0,
        "robot_radius": 0.3
    },
    prediction_horizon=3.0  # secondi
)

# Simula navigazione
current_pose = start_pose
for i, waypoint in enumerate(global_path.waypoints):
    # Ottieni dati sensori
    lidar_data = await robot_controller.get_lidar_data()
    odometry = await robot_controller.get_odometry()
    
    # Aggiorna SLAM
    await slam_system.update(lidar_data, odometry)
    
    # Pianifica movimento locale
    local_goal = waypoint
    velocity_command = await local_planner.compute_velocity_command(
        current_pose,
        local_goal,
        lidar_data
    )
    
    # Esegui comando di velocità
    await robot_controller.execute_velocity_command(velocity_command)
    
    # Aggiorna posizione
    current_pose = await robot_controller.get_current_pose()
    
    print(f"Waypoint {i}: Position ({current_pose['x']:.2f}, {current_pose['y']:.2f})")
    
    await asyncio.sleep(0.1)

# Evitamento ostacoli dinamici
obstacle_avoidance = ObstacleAvoidance(
    algorithm="artificial_potential_field",  # "apf", "vector_field", "social_force"
    safety_distance=0.5,
    max_avoidance_force=2.0
)

# Rileva e evita ostacoli
while True:
    lidar_data = await robot_controller.get_lidar_data()
    dynamic_obstacles = await obstacle_avoidance.detect_dynamic_obstacles(lidar_data)
    
    if dynamic_obstacles:
        avoidance_velocity = await obstacle_avoidance.compute_avoidance_velocity(
            current_pose,
            goal_pose,
            dynamic_obstacles
        )
        await robot_controller.execute_velocity_command(avoidance_velocity)
    
    await asyncio.sleep(0.05)
```

### Manipulation

```python
from src.applications.robotics.manipulation import (
    GraspPlanner, MotionPlanner, ForceController, VisionGuidedManipulation
)

# Pianificatore di presa
grasp_planner = GraspPlanner(
    gripper_type="parallel_jaw",  # "parallel_jaw", "suction", "multi_finger"
    grasp_database="grasp_db.json",
    quality_metrics=["force_closure", "stability", "reachability"]
)

# Carica database di prese
await grasp_planner.load_grasp_database()

# Rileva oggetti nella scena
from src.multimodal.vision.object_detection import YOLODetector

object_detector = YOLODetector(model_name="yolov8n")
await object_detector.load_model()

# Cattura immagine dalla camera del robot
camera_image = await robot_controller.capture_camera_image()
objects = await object_detector.detect(camera_image)

print(f"Detected {len(objects)} objects in scene")

# Pianifica presa per ogni oggetto
for obj in objects:
    if obj.class_name in ["bottle", "cup", "box"]:
        # Stima pose 3D dell'oggetto
        object_pose = await robot_controller.estimate_object_pose(
            obj.bounding_box,
            obj.class_name
        )
        
        # Genera candidate di presa
        grasp_candidates = await grasp_planner.generate_grasps(
            object_pose,
            obj.class_name
        )
        
        # Valuta qualità delle prese
        best_grasp = await grasp_planner.select_best_grasp(
            grasp_candidates,
            current_robot_pose=await robot_controller.get_current_pose()
        )
        
        if best_grasp:
            print(f"Best grasp for {obj.class_name}: quality = {best_grasp.quality:.3f}")
            
            # Pianifica movimento per raggiungere la presa
            motion_planner = MotionPlanner(
                robot_model=robot_controller.robot_model,
                collision_checking=True
            )
            
            # Pre-grasp pose
            pre_grasp_pose = best_grasp.pose.copy()
            pre_grasp_pose["position"][2] += 0.1  # 10cm sopra
            
            # Pianifica traiettoria
            trajectory = await motion_planner.plan_trajectory(
                start_pose=await robot_controller.get_current_pose(),
                goal_pose=pre_grasp_pose,
                intermediate_poses=[pre_grasp_pose, best_grasp.pose]
            )
            
            if trajectory.success:
                # Esegui movimento
                await robot_controller.execute_trajectory(trajectory)
                
                # Controllo di forza per la presa
                force_controller = ForceController(
                    max_force=50.0,  # Newton
                    force_threshold=10.0,
                    compliance_mode=True
                )
                
                # Avvicinamento con controllo di forza
                await force_controller.approach_with_force_control(
                    robot_controller,
                    target_pose=best_grasp.pose,
                    approach_velocity=0.05  # m/s
                )
                
                # Chiudi gripper
                await robot_controller.close_gripper(force=best_grasp.force)
                
                # Verifica successo presa
                grasp_success = await robot_controller.check_grasp_success()
                
                if grasp_success:
                    print(f"Successfully grasped {obj.class_name}")
                    
                    # Solleva oggetto
                    lift_pose = best_grasp.pose.copy()
                    lift_pose["position"][2] += 0.2
                    await robot_controller.move_to_pose(lift_pose)
                else:
                    print(f"Failed to grasp {obj.class_name}")
                    await robot_controller.open_gripper()

# Manipolazione guidata dalla visione
vision_manipulation = VisionGuidedManipulation(
    camera_config={
        "intrinsic_matrix": np.array([[525, 0, 320], [0, 525, 240], [0, 0, 1]]),
        "distortion_coeffs": np.array([0.1, -0.2, 0, 0, 0])
    },
    hand_eye_calibration="eye_in_hand"  # "eye_in_hand", "eye_to_hand"
)

# Calibrazione hand-eye
calibration_poses = []
calibration_images = []

for i in range(10):
    # Muovi robot in pose casuali
    random_pose = await robot_controller.generate_random_pose()
    await robot_controller.move_to_pose(random_pose)
    
    # Cattura immagine e pose
    image = await robot_controller.capture_camera_image()
    pose = await robot_controller.get_current_pose()
    
    calibration_poses.append(pose)
    calibration_images.append(image)

# Esegui calibrazione
calibration_result = await vision_manipulation.calibrate_hand_eye(
    calibration_poses,
    calibration_images
)

if calibration_result.success:
    print(f"Hand-eye calibration successful. Error: {calibration_result.reprojection_error:.3f}")
    
    # Usa calibrazione per pick-and-place guidato dalla visione
    target_object = "red_block"
    target_location = {"x": 0.5, "y": -0.2, "z": 0.1}
    
    success = await vision_manipulation.pick_and_place(
        robot_controller,
        target_object,
        target_location
    )
    
    if success:
        print(f"Successfully placed {target_object} at target location")
    else:
        print(f"Failed to complete pick-and-place task")
else:
    print(f"Hand-eye calibration failed: {calibration_result.error_message}")
```

## Industrial

### Process Control

```python
from src.applications.industrial.process_control import (
    PIDController, ModelPredictiveController, ProcessMonitor, ControlLoop
)

# Controller PID per controllo temperatura
temperature_controller = PIDController(
    kp=2.0,
    ki=0.5,
    kd=0.1,
    setpoint=75.0,  # °C
    output_limits=(-100, 100),
    sample_time=1.0  # secondi
)

# Monitor del processo
process_monitor = ProcessMonitor(
    variables=[
        {"name": "temperature", "unit": "°C", "range": (0, 150)},
        {"name": "pressure", "unit": "bar", "range": (0, 10)},
        {"name": "flow_rate", "unit": "L/min", "range": (0, 100)}
    ],
    sampling_rate=1.0,  # Hz
    data_retention=24*3600  # 24 ore
)

# Avvia monitoring
await process_monitor.start()

# Loop di controllo
control_loop = ControlLoop(
    controller=temperature_controller,
    monitor=process_monitor,
    control_variable="temperature",
    output_variable="heater_power"
)

# Simula controllo processo
for i in range(1000):
    # Leggi valore corrente
    current_temp = await process_monitor.read_variable("temperature")
    
    # Calcola output del controller
    control_output = temperature_controller.update(current_temp)
    
    # Applica controllo
    await process_monitor.write_variable("heater_power", control_output)
    
    # Log dati
    await process_monitor.log_data({
        "timestamp": time.time(),
        "temperature": current_temp,
        "setpoint": temperature_controller.setpoint,
        "control_output": control_output
    })
    
    await asyncio.sleep(1.0)

# Controller predittivo per processi complessi
mpc_controller = ModelPredictiveController(
    model_type="linear",  # "linear", "nonlinear", "neural_network"
    prediction_horizon=10,
    control_horizon=3,
    constraints={
        "input_min": [-50, -30],
        "input_max": [50, 30],
        "output_min": [60, 0.5],
        "output_max": [90, 5.0]
    },
    weights={
        "output": [1.0, 0.5],
        "input": [0.1, 0.1],
        "input_rate": [0.01, 0.01]
    }
)

# Identifica modello del processo
identification_data = await process_monitor.get_historical_data(
    variables=["temperature", "pressure", "heater_power", "valve_position"],
    duration=3600  # 1 ora
)

model = await mpc_controller.identify_model(
    inputs=["heater_power", "valve_position"],
    outputs=["temperature", "pressure"],
    data=identification_data
)

print(f"Model identified with R² = {model.r_squared:.3f}")

# Usa MPC per controllo multivariabile
setpoints = {"temperature": 80.0, "pressure": 2.5}

for i in range(100):
    # Stato corrente
    current_state = {
        "temperature": await process_monitor.read_variable("temperature"),
        "pressure": await process_monitor.read_variable("pressure")
    }
    
    # Calcola controllo ottimale
    optimal_inputs = await mpc_controller.compute_control(
        current_state,
        setpoints
    )
    
    # Applica controlli
    await process_monitor.write_variable("heater_power", optimal_inputs["heater_power"])
    await process_monitor.write_variable("valve_position", optimal_inputs["valve_position"])
    
    await asyncio.sleep(1.0)
```

### Quality Control

```python
from src.applications.industrial.quality_control import (
    DefectDetector, QualityInspector, StatisticalProcessControl, QualityMetrics
)

# Rilevatore di difetti basato su visione
defect_detector = DefectDetector(
    model_type="anomaly_detection",  # "classification", "segmentation", "anomaly_detection"
    model_path="defect_detection_model.pt",
    confidence_threshold=0.8,
    defect_classes=["scratch", "dent", "discoloration", "crack"]
)

await defect_detector.load_model()

# Ispettore qualità
quality_inspector = QualityInspector(
    inspection_stations=[
        {"name": "visual_inspection", "type": "camera", "detector": defect_detector},
        {"name": "dimensional_check", "type": "laser_scanner"},
        {"name": "surface_roughness", "type": "profilometer"}
    ],
    quality_standards={
        "defect_tolerance": 0.05,  # 5% difetti accettabili
        "dimensional_tolerance": 0.1,  # ±0.1mm
        "surface_roughness_max": 1.6  # Ra μm
    }
)

# Controllo statistico del processo
spc = StatisticalProcessControl(
    control_charts=["x_bar", "r_chart", "p_chart"],
    sample_size=5,
    control_limits_sigma=3
)

# Simula ispezione qualità
production_batch = []
quality_data = []

for part_id in range(100):
    # Cattura immagine del pezzo
    part_image = await quality_inspector.capture_image(f"part_{part_id}")
    
    # Ispezione visiva
    visual_result = await defect_detector.inspect(part_image)
    
    # Controllo dimensionale
    dimensions = await quality_inspector.measure_dimensions(part_id)
    
    # Controllo rugosità superficie
    surface_roughness = await quality_inspector.measure_surface_roughness(part_id)
    
    # Valuta qualità complessiva
    quality_result = await quality_inspector.evaluate_quality({
        "visual": visual_result,
        "dimensions": dimensions,
        "surface_roughness": surface_roughness
    })
    
    # Registra dati
    quality_data.append({
        "part_id": part_id,
        "timestamp": time.time(),
        "defects_found": len(visual_result.defects),
        "dimensional_deviation": abs(dimensions["length"] - 100.0),  # target 100mm
        "surface_roughness": surface_roughness,
        "overall_quality": quality_result.quality_score,
        "pass_fail": quality_result.passed
    })
    
    production_batch.append(quality_result)
    
    # Aggiorna controllo statistico
    if len(quality_data) >= 5:  # Campione di 5 pezzi
        sample_data = quality_data[-5:]
        spc_result = await spc.update_control_charts(sample_data)
        
        if spc_result.out_of_control:
            print(f"ALERT: Process out of control at part {part_id}")
            print(f"Control chart violations: {spc_result.violations}")
            
            # Azioni correttive
            await quality_inspector.trigger_corrective_action(spc_result)
    
    print(f"Part {part_id}: Quality = {quality_result.quality_score:.2f}, "
          f"Status = {'PASS' if quality_result.passed else 'FAIL'}")

# Analisi qualità batch
quality_metrics = QualityMetrics()
batch_analysis = quality_metrics.analyze_batch(production_batch)

print(f"\nBatch Quality Analysis:")
print(f"  Total parts: {batch_analysis.total_parts}")
print(f"  Pass rate: {batch_analysis.pass_rate:.1%}")
print(f"  Average quality score: {batch_analysis.avg_quality_score:.2f}")
print(f"  Defect rate: {batch_analysis.defect_rate:.1%}")
print(f"  Most common defect: {batch_analysis.most_common_defect}")

# Genera report qualità
quality_report = await quality_inspector.generate_quality_report(
    batch_data=quality_data,
    time_period="last_24_hours"
)

print(f"\nQuality Report Generated: {quality_report.filename}")
print(f"Report includes: {', '.join(quality_report.sections)}")
```

### Predictive Maintenance

```python
from src.applications.industrial.predictive_maintenance import (
    ConditionMonitor, FailurePrediction, MaintenanceScheduler, AssetManager
)

# Monitor delle condizioni
condition_monitor = ConditionMonitor(
    sensors=[
        {"name": "vibration_x", "type": "accelerometer", "sampling_rate": 1000},
        {"name": "vibration_y", "type": "accelerometer", "sampling_rate": 1000},
        {"name": "vibration_z", "type": "accelerometer", "sampling_rate": 1000},
        {"name": "temperature", "type": "thermocouple", "sampling_rate": 1},
        {"name": "current", "type": "current_sensor", "sampling_rate": 10},
        {"name": "acoustic", "type": "microphone", "sampling_rate": 44100}
    ],
    feature_extraction=[
        "rms", "peak", "crest_factor", "kurtosis", "skewness",
        "spectral_centroid", "spectral_rolloff", "mfcc"
    ]
)

# Avvia monitoring
await condition_monitor.start()

# Predittore di guasti
failure_predictor = FailurePrediction(
    model_type="lstm",  # "lstm", "transformer", "isolation_forest", "svm"
    prediction_horizon=168,  # ore (1 settimana)
    confidence_threshold=0.8,
    failure_modes=[
        "bearing_failure", "motor_overheating", "belt_wear",
        "misalignment", "imbalance", "looseness"
    ]
)

# Addestra modello su dati storici
historical_data = await condition_monitor.get_historical_data(
    start_date="2023-01-01",
    end_date="2023-12-31"
)

training_result = await failure_predictor.train_model(
    historical_data,
    validation_split=0.2
)

print(f"Model trained with accuracy: {training_result.accuracy:.3f}")
print(f"Precision: {training_result.precision:.3f}")
print(f"Recall: {training_result.recall:.3f}")

# Gestore asset
asset_manager = AssetManager(
    assets=[
        {
            "id": "PUMP_001",
            "type": "centrifugal_pump",
            "criticality": "high",
            "sensors": ["vibration_x", "vibration_y", "temperature", "current"]
        },
        {
            "id": "MOTOR_001",
            "type": "induction_motor",
            "criticality": "medium",
            "sensors": ["vibration_z", "temperature", "current", "acoustic"]
        }
    ]
)

# Scheduler manutenzione
maintenance_scheduler = MaintenanceScheduler(
    strategies={
        "preventive": {"interval": 30*24, "cost": 1000},  # 30 giorni
        "predictive": {"threshold": 0.7, "cost": 1500},
        "corrective": {"cost": 5000, "downtime": 24}  # ore
    },
    optimization_objective="minimize_cost"  # "minimize_cost", "minimize_downtime"
)

# Loop di monitoraggio predittivo
while True:
    for asset in asset_manager.assets:
        # Raccoglie dati sensori
        sensor_data = await condition_monitor.collect_sensor_data(
            asset_id=asset["id"],
            duration=3600  # 1 ora
        )
        
        # Estrai features
        features = await condition_monitor.extract_features(sensor_data)
        
        # Predici guasti
        prediction = await failure_predictor.predict_failure(
            features,
            asset_type=asset["type"]
        )
        
        if prediction.failure_probability > 0.7:
            print(f"HIGH RISK: {asset['id']} - {prediction.failure_mode}")
            print(f"Probability: {prediction.failure_probability:.3f}")
            print(f"Estimated time to failure: {prediction.time_to_failure:.1f} hours")
            
            # Pianifica manutenzione predittiva
            maintenance_plan = await maintenance_scheduler.schedule_maintenance(
                asset_id=asset["id"],
                failure_prediction=prediction,
                current_workload=await asset_manager.get_current_workload(asset["id"])
            )
            
            print(f"Maintenance scheduled: {maintenance_plan.scheduled_date}")
            print(f"Estimated cost: ${maintenance_plan.estimated_cost}")
            print(f"Expected downtime: {maintenance_plan.expected_downtime} hours")
            
            # Notifica team manutenzione
            await maintenance_scheduler.notify_maintenance_team(maintenance_plan)
        
        elif prediction.failure_probability > 0.4:
            print(f"MEDIUM RISK: {asset['id']} - monitoring closely")
            
            # Aumenta frequenza monitoraggio
            await condition_monitor.increase_monitoring_frequency(
                asset_id=asset["id"],
                factor=2.0
            )
        
        # Aggiorna stato asset
        await asset_manager.update_asset_condition(
            asset_id=asset["id"],
            condition_score=1.0 - prediction.failure_probability,
            last_prediction=prediction
        )
    
    await asyncio.sleep(3600)  # Controlla ogni ora

# Analisi ROI manutenzione predittiva
roi_analysis = await maintenance_scheduler.calculate_roi(
    time_period="last_year",
    baseline="preventive_only"
)

print(f"\nPredictive Maintenance ROI Analysis:")
print(f"  Cost savings: ${roi_analysis.cost_savings:,.2f}")
print(f"  Downtime reduction: {roi_analysis.downtime_reduction:.1f} hours")
print(f"  ROI: {roi_analysis.roi:.1%}")
print(f"  Payback period: {roi_analysis.payback_period:.1f} months")
```

## Edge AI

### Edge Deployment

```python
from src.applications.edge_ai.edge_deployment import (
    EdgeDevice, ModelOptimizer, EdgeInference, ResourceManager
)

# Configurazione dispositivo edge
edge_device = EdgeDevice(
    device_type="nvidia_jetson_nano",  # "raspberry_pi", "intel_nuc", "custom"
    specs={
        "cpu_cores": 4,
        "ram_gb": 4,
        "gpu_memory_mb": 2048,
        "storage_gb": 64,
        "power_budget_watts": 10
    },
    connectivity=["wifi", "ethernet", "4g"]
)

# Ottimizzatore modelli
model_optimizer = ModelOptimizer(
    target_device=edge_device,
    optimization_techniques=[
        "quantization",      # INT8/FP16
        "pruning",          # Rimozione pesi
        "knowledge_distillation",  # Modello teacher-student
        "tensorrt_optimization"     # NVIDIA TensorRT
    ]
)

# Ottimizza modello per edge
original_model_path = "large_model.pt"
optimized_model = await model_optimizer.optimize_model(
    model_path=original_model_path,
    target_latency_ms=100,
    target_accuracy_drop=0.02,  # Max 2% accuracy drop
    optimization_level="aggressive"  # "conservative", "moderate", "aggressive"
)

print(f"Model optimization results:")
print(f"  Original size: {optimized_model.original_size_mb:.1f} MB")
print(f"  Optimized size: {optimized_model.optimized_size_mb:.1f} MB")
print(f"  Size reduction: {optimized_model.size_reduction:.1%}")
print(f"  Latency improvement: {optimized_model.latency_improvement:.1%}")
print(f"  Accuracy drop: {optimized_model.accuracy_drop:.3f}")

# Deploy su dispositivo edge
deployment_result = await edge_device.deploy_model(
    optimized_model,
    deployment_config={
        "auto_start": True,
        "health_check_interval": 60,
        "max_memory_usage": 0.8,
        "fallback_model": "lightweight_backup.pt"
    }
)

if deployment_result.success:
    print(f"Model deployed successfully to {edge_device.device_id}")
else:
    print(f"Deployment failed: {deployment_result.error_message}")

# Inference engine per edge
edge_inference = EdgeInference(
    device=edge_device,
    model=optimized_model,
    batch_size=1,  # Tipicamente 1 per edge
    preprocessing_pipeline=[
        "resize", "normalize", "to_tensor"
    ]
)

# Gestore risorse
resource_manager = ResourceManager(
    device=edge_device,
    monitoring_interval=10,  # secondi
    resource_limits={
        "cpu_usage": 0.8,
        "memory_usage": 0.9,
        "gpu_usage": 0.85,
        "temperature": 70  # °C
    }
)

# Avvia monitoring risorse
await resource_manager.start_monitoring()

# Simula inference edge
for i in range(1000):
    # Simula input data
    input_data = await edge_device.get_sensor_data()
    
    # Controlla risorse disponibili
    resources = await resource_manager.get_current_resources()
    
    if resources.cpu_usage < 0.8 and resources.memory_usage < 0.9:
        # Esegui inference
        start_time = time.time()
        result = await edge_inference.predict(input_data)
        inference_time = time.time() - start_time
        
        print(f"Inference {i}: {result.prediction} (confidence: {result.confidence:.3f}, "
              f"time: {inference_time*1000:.1f}ms)")
        
        # Log metriche
        await resource_manager.log_inference_metrics({
            "inference_id": i,
            "latency_ms": inference_time * 1000,
            "confidence": result.confidence,
            "cpu_usage": resources.cpu_usage,
            "memory_usage": resources.memory_usage,
            "gpu_usage": resources.gpu_usage
        })
    else:
        print(f"Skipping inference {i}: Resource constraints")
        await asyncio.sleep(1.0)
    
    await asyncio.sleep(0.1)
```

### Federated Learning

```python
from src.applications.edge_ai.federated_learning import (
    FederatedClient, FederatedServer, ModelAggregator, PrivacyPreserver
)

# Client federato
federated_client = FederatedClient(
    client_id="edge_device_001",
    device_specs=edge_device.specs,
    local_data_size=1000,
    privacy_level="high"  # "low", "medium", "high"
)

# Preservazione privacy
privacy_preserver = PrivacyPreserver(
    techniques=[
        "differential_privacy",
        "secure_aggregation",
        "homomorphic_encryption"
    ],
    privacy_budget=1.0,  # epsilon per differential privacy
    noise_multiplier=1.1
)

# Training locale
local_model = await federated_client.get_local_model()
local_data = await federated_client.get_local_data()

# Training con privacy
for round_num in range(10):
    print(f"Federated Learning Round {round_num + 1}")
    
    # Ricevi modello globale dal server
    global_model = await federated_client.receive_global_model()
    
    # Training locale
    local_updates = await federated_client.train_local_model(
        global_model,
        local_data,
        epochs=5,
        learning_rate=0.01
    )
    
    # Applica privacy preservation
    private_updates = await privacy_preserver.apply_privacy(
        local_updates,
        privacy_technique="differential_privacy"
    )
    
    # Invia aggiornamenti al server
    await federated_client.send_model_updates(private_updates)
    
    # Valuta modello locale
    local_accuracy = await federated_client.evaluate_local_model()
    print(f"  Local accuracy: {local_accuracy:.3f}")

# Server federato (simulazione)
federated_server = FederatedServer(
    aggregation_strategy="fedavg",  # "fedavg", "fedprox", "scaffold"
    min_clients=5,
    max_clients=100,
    rounds=50
)

# Aggregatore modelli
model_aggregator = ModelAggregator(
    aggregation_method="weighted_average",
    weight_strategy="data_size"  # "data_size", "uniform", "performance"
)

# Simula aggregazione (lato server)
client_updates = [
    {"client_id": f"client_{i}", "updates": torch.randn(1000), "data_size": 500 + i*100}
    for i in range(10)
]

aggregated_model = await model_aggregator.aggregate_updates(
    client_updates,
    current_global_model=global_model
)

print(f"Model aggregated from {len(client_updates)} clients")

# Valuta modello globale
global_accuracy = await federated_server.evaluate_global_model(
    aggregated_model,
    test_data
)

print(f"Global model accuracy: {global_accuracy:.3f}")
```

## Healthcare

### Medical Imaging

```python
from src.applications.healthcare.medical_imaging import (
    MedicalImageProcessor, DiagnosticAI, ImageSegmentation, ReportGenerator
)

# Processore immagini mediche
medical_processor = MedicalImageProcessor(
    modalities=["xray", "ct", "mri", "ultrasound"],
    preprocessing_steps=[
        "dicom_conversion",
        "intensity_normalization",
        "noise_reduction",
        "contrast_enhancement"
    ],
    anonymization=True
)

# AI diagnostica
diagnostic_ai = DiagnosticAI(
    models={
        "chest_xray": "chest_xray_classifier.pt",
        "brain_mri": "brain_tumor_detector.pt",
        "ct_lung": "lung_nodule_detector.pt"
    },
    confidence_threshold=0.8,
    ensemble_voting=True
)

await diagnostic_ai.load_models()

# Segmentazione immagini mediche
medical_segmentation = ImageSegmentation(
    models={
        "organ_segmentation": "unet_organ_seg.pt",
        "tumor_segmentation": "nnunet_tumor_seg.pt",
        "vessel_segmentation": "vessel_seg_model.pt"
    },
    post_processing=[
        "morphological_operations",
        "connected_components",
        "hole_filling"
    ]
)

# Analizza immagine medica
dicom_file = "patient_001_chest_xray.dcm"

# Preprocessa immagine
processed_image = await medical_processor.process_dicom(
    dicom_file,
    modality="xray",
    body_part="chest"
)

print(f"Processed image shape: {processed_image.shape}")
print(f"Patient ID: {processed_image.metadata['patient_id']}")
print(f"Study date: {processed_image.metadata['study_date']}")

# Diagnosi AI
diagnosis_result = await diagnostic_ai.diagnose(
    processed_image,
    modality="chest_xray"
)

print(f"\nDiagnosis Results:")
for finding in diagnosis_result.findings:
    print(f"  {finding.condition}: {finding.confidence:.3f}")
    if finding.confidence > 0.8:
        print(f"    Location: {finding.location}")
        print(f"    Severity: {finding.severity}")

# Segmentazione anatomica
segmentation_result = await medical_segmentation.segment(
    processed_image,
    task="organ_segmentation"
)

print(f"\nSegmentation Results:")
for organ, mask in segmentation_result.masks.items():
    volume = np.sum(mask) * processed_image.metadata['pixel_spacing']
    print(f"  {organ}: {volume:.2f} cm³")

# Genera report medico
report_generator = ReportGenerator(
    template_type="radiology",
    language="italian",
    include_ai_confidence=True
)

medical_report = await report_generator.generate_report(
    patient_info=processed_image.metadata,
    diagnosis_results=diagnosis_result,
    segmentation_results=segmentation_result,
    radiologist_notes="Paziente con sintomi respiratori"
)

print(f"\nMedical Report Generated:")
print(f"Report ID: {medical_report.report_id}")
print(f"Generated at: {medical_report.timestamp}")
print(f"Confidence level: {medical_report.overall_confidence:.3f}")

# Salva report
await report_generator.save_report(
    medical_report,
    format="pdf",
    output_path="reports/"
)
```

### Clinical Decision Support

```python
from src.applications.healthcare.clinical_decision_support import (
    ClinicalDSS, RiskAssessment, TreatmentRecommendation, DrugInteractionChecker
)

# Sistema di supporto decisionale clinico
clinical_dss = ClinicalDSS(
    knowledge_base="medical_kb.json",
    guidelines=["who_guidelines", "aha_guidelines", "esc_guidelines"],
    evidence_level_threshold="B"  # A, B, C
)

# Valutazione del rischio
risk_assessment = RiskAssessment(
    risk_models={
        "cardiovascular": "framingham_risk_score",
        "diabetes": "diabetes_risk_calculator",
        "stroke": "chads2_vasc_score"
    }
)

# Raccomandazioni terapeutiche
treatment_recommender = TreatmentRecommendation(
    treatment_database="treatment_protocols.db",
    personalization_factors=[
        "age", "gender", "comorbidities", "allergies", "current_medications"
    ]
)

# Checker interazioni farmacologiche
drug_checker = DrugInteractionChecker(
    drug_database="drug_interactions.db",
    severity_levels=["minor", "moderate", "major", "contraindicated"]
)

# Caso clinico
patient_data = {
    "patient_id": "P001",
    "age": 65,
    "gender": "male",
    "weight": 80,  # kg
    "height": 175,  # cm
    "symptoms": ["chest_pain", "shortness_of_breath", "fatigue"],
    "vital_signs": {
        "blood_pressure": "150/95",
        "heart_rate": 85,
        "temperature": 36.8,
        "oxygen_saturation": 96
    },
    "lab_results": {
        "cholesterol_total": 240,  # mg/dL
        "ldl_cholesterol": 160,
        "hdl_cholesterol": 35,
        "triglycerides": 200,
        "glucose": 110,
        "creatinine": 1.2
    },
    "medical_history": ["hypertension", "diabetes_type2"],
    "current_medications": ["metformin", "lisinopril"],
    "allergies": ["penicillin"]
}

# Valutazione rischio cardiovascolare
cardiovascular_risk = await risk_assessment.assess_cardiovascular_risk(
    patient_data
)

print(f"Cardiovascular Risk Assessment:")
print(f"  10-year risk: {cardiovascular_risk.ten_year_risk:.1%}")
print(f"  Risk category: {cardiovascular_risk.risk_category}")
print(f"  Risk factors: {', '.join(cardiovascular_risk.risk_factors)}")

# Raccomandazioni terapeutiche
treatment_recommendations = await treatment_recommender.get_recommendations(
    patient_data,
    primary_diagnosis="acute_coronary_syndrome",
    evidence_level="A"
)

print(f"\nTreatment Recommendations:")
for recommendation in treatment_recommendations:
    print(f"  {recommendation.intervention}:")
    print(f"    Strength: {recommendation.strength}")
    print(f"    Evidence level: {recommendation.evidence_level}")
    print(f"    Dosage: {recommendation.dosage}")
    print(f"    Duration: {recommendation.duration}")

# Controllo interazioni farmacologiche
proposed_medications = ["aspirin", "atorvastatin", "metoprolol"]
interaction_check = await drug_checker.check_interactions(
    current_medications=patient_data["current_medications"],
    proposed_medications=proposed_medications,
    patient_allergies=patient_data["allergies"]
)

print(f"\nDrug Interaction Check:")
if interaction_check.interactions:
    for interaction in interaction_check.interactions:
        print(f"  {interaction.drug1} + {interaction.drug2}:")
        print(f"    Severity: {interaction.severity}")
        print(f"    Description: {interaction.description}")
        print(f"    Management: {interaction.management}")
else:
    print("  No significant interactions found")

# Supporto decisionale integrato
decision_support = await clinical_dss.provide_decision_support(
    patient_data=patient_data,
    clinical_question="optimal_treatment_acute_mi",
    context={
        "risk_assessment": cardiovascular_risk,
        "treatment_recommendations": treatment_recommendations,
        "drug_interactions": interaction_check
    }
)

print(f"\nClinical Decision Support:")
print(f"  Recommended action: {decision_support.primary_recommendation}")
print(f"  Confidence: {decision_support.confidence:.3f}")
print(f"  Supporting evidence: {decision_support.evidence_summary}")
print(f"  Monitoring requirements: {', '.join(decision_support.monitoring_plan)}")

# Alert clinici
if decision_support.alerts:
    print(f"\nClinical Alerts:")
    for alert in decision_support.alerts:
        print(f"  {alert.severity}: {alert.message}")
```

## Financial

### Algorithmic Trading

```python
from src.applications.financial.algorithmic_trading import (
    TradingStrategy, RiskManager, PortfolioOptimizer, MarketDataFeed
)

# Feed dati di mercato
market_feed = MarketDataFeed(
    data_sources=["yahoo_finance", "alpha_vantage", "quandl"],
    symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
    timeframe="1min",
    real_time=True
)

# Strategia di trading
trading_strategy = TradingStrategy(
    strategy_type="momentum",  # "momentum", "mean_reversion", "arbitrage"
    parameters={
        "lookback_period": 20,
        "momentum_threshold": 0.02,
        "stop_loss": 0.05,
        "take_profit": 0.10
    },
    universe=["AAPL", "GOOGL", "MSFT"]
)

# Gestore del rischio
risk_manager = RiskManager(
    max_position_size=0.1,  # 10% del portafoglio
    max_daily_loss=0.02,    # 2% perdita giornaliera
    max_drawdown=0.15,      # 15% drawdown massimo
    var_confidence=0.95,    # 95% VaR
    stress_scenarios=[
        "market_crash_2008",
        "covid_crash_2020",
        "dot_com_bubble_2000"
    ]
)

# Ottimizzatore portafoglio
portfolio_optimizer = PortfolioOptimizer(
    optimization_method="mean_variance",  # "mean_variance", "risk_parity", "black_litterman"
    constraints={
        "max_weight": 0.3,
        "min_weight": 0.0,
        "sector_limits": {"tech": 0.5, "finance": 0.3}
    },
    rebalancing_frequency="weekly"
)

# Avvia feed dati
await market_feed.start()

# Loop di trading
portfolio_value = 100000  # $100k iniziali
positions = {}
trade_history = []

while True:
    # Ottieni dati di mercato
    market_data = await market_feed.get_latest_data()
    
    # Calcola segnali di trading
    signals = await trading_strategy.generate_signals(market_data)
    
    for symbol, signal in signals.items():
        if signal.action in ["BUY", "SELL"]:
            # Valutazione del rischio
            risk_assessment = await risk_manager.assess_trade_risk(
                symbol=symbol,
                action=signal.action,
                quantity=signal.quantity,
                current_portfolio=positions,
                market_data=market_data
            )
            
            if risk_assessment.approved:
                # Esegui trade
                trade_result = await execute_trade(
                    symbol=symbol,
                    action=signal.action,
                    quantity=risk_assessment.adjusted_quantity,
                    price=market_data[symbol]["price"]
                )
                
                if trade_result.success:
                    # Aggiorna posizioni
                    if symbol not in positions:
                        positions[symbol] = 0
                    
                    if signal.action == "BUY":
                        positions[symbol] += trade_result.quantity
                    else:
                        positions[symbol] -= trade_result.quantity
                    
                    # Registra trade
                    trade_history.append({
                        "timestamp": time.time(),
                        "symbol": symbol,
                        "action": signal.action,
                        "quantity": trade_result.quantity,
                        "price": trade_result.price,
                        "commission": trade_result.commission
                    })
                    
                    print(f"Trade executed: {signal.action} {trade_result.quantity} {symbol} @ ${trade_result.price}")
            else:
                print(f"Trade rejected: {symbol} - {risk_assessment.rejection_reason}")
    
    # Ottimizzazione portafoglio (settimanale)
    if len(trade_history) % 1000 == 0:  # Ogni 1000 iterazioni
        optimal_weights = await portfolio_optimizer.optimize_portfolio(
            current_positions=positions,
            market_data=market_data,
            expected_returns=await calculate_expected_returns(market_data),
            risk_model=await risk_manager.get_risk_model()
        )
        
        print(f"Portfolio optimization completed:")
        for symbol, weight in optimal_weights.items():
            print(f"  {symbol}: {weight:.2%}")
    
    await asyncio.sleep(60)  # Attendi 1 minuto

# Analisi performance
performance_metrics = await calculate_performance_metrics(
    trade_history,
    initial_capital=100000
)

print(f"\nTrading Performance:")
print(f"  Total return: {performance_metrics.total_return:.2%}")
print(f"  Sharpe ratio: {performance_metrics.sharpe_ratio:.3f}")
print(f"  Max drawdown: {performance_metrics.max_drawdown:.2%}")
print(f"  Win rate: {performance_metrics.win_rate:.1%}")
print(f"  Profit factor: {performance_metrics.profit_factor:.2f}")
```

### Risk Management

```python
from src.applications.financial.risk_management import (
    VaRCalculator, StressTestEngine, CreditRiskModel, LiquidityRiskAssessment
)

# Calcolatore Value at Risk
var_calculator = VaRCalculator(
    methods=["historical", "parametric", "monte_carlo"],
    confidence_levels=[0.95, 0.99, 0.999],
    holding_period=1  # giorni
)

# Engine stress test
stress_engine = StressTestEngine(
    scenarios=[
        "interest_rate_shock",
        "equity_market_crash",
        "credit_spread_widening",
        "currency_devaluation",
        "liquidity_crisis"
    ],
    severity_levels=["mild", "moderate", "severe", "extreme"]
)

# Modello rischio credito
credit_model = CreditRiskModel(
    model_type="merton",  # "merton", "reduced_form", "machine_learning"
    rating_system="internal",
    pd_horizon=1  # anni
)

# Valutazione rischio liquidità
liquidity_assessment = LiquidityRiskAssessment(
    metrics=["bid_ask_spread", "market_impact", "time_to_liquidate"],
    market_conditions="normal"  # "normal", "stressed", "crisis"
)

# Portfolio per analisi rischio
portfolio = {
    "equities": {
        "AAPL": {"quantity": 100, "price": 150, "beta": 1.2},
        "GOOGL": {"quantity": 50, "price": 2500, "beta": 1.1},
        "MSFT": {"quantity": 75, "price": 300, "beta": 0.9}
    },
    "bonds": {
        "US10Y": {"quantity": 1000, "price": 95, "duration": 8.5},
        "CORP_AAA": {"quantity": 500, "price": 98, "duration": 5.2}
    },
    "derivatives": {
        "SPY_CALL": {"quantity": 10, "delta": 0.6, "gamma": 0.03, "vega": 0.15}
    }
}

# Calcola VaR
historical_data = await get_historical_returns(portfolio, days=252)
var_results = await var_calculator.calculate_var(
    portfolio=portfolio,
    historical_data=historical_data
)

print(f"Value at Risk Analysis:")
for method, results in var_results.items():
    print(f"  {method.upper()} VaR:")
    for confidence, var_value in results.items():
        print(f"    {confidence:.1%} confidence: ${var_value:,.2f}")

# Stress testing
stress_results = await stress_engine.run_stress_tests(
    portfolio=portfolio,
    scenarios=["equity_market_crash", "interest_rate_shock"]
)

print(f"\nStress Test Results:")
for scenario, results in stress_results.items():
    print(f"  {scenario}:")
    print(f"    Portfolio loss: ${results.portfolio_loss:,.2f}")
    print(f"    Loss percentage: {results.loss_percentage:.2%}")
    print(f"    Worst asset: {results.worst_performing_asset}")

# Analisi rischio credito
credit_exposures = {
    "CORP_001": {"exposure": 1000000, "rating": "BBB", "sector": "technology"},
    "CORP_002": {"exposure": 500000, "rating": "A", "sector": "healthcare"},
    "CORP_003": {"exposure": 750000, "rating": "BB", "sector": "energy"}
}

credit_risk_results = await credit_model.assess_credit_risk(credit_exposures)

print(f"\nCredit Risk Analysis:")
for entity, risk in credit_risk_results.items():
    print(f"  {entity}:")
    print(f"    PD (1Y): {risk.probability_of_default:.2%}")
    print(f"    LGD: {risk.loss_given_default:.2%}")
    print(f"    Expected Loss: ${risk.expected_loss:,.2f}")

# Valutazione rischio liquidità
liquidity_results = await liquidity_assessment.assess_liquidity_risk(
    portfolio=portfolio,
    liquidation_horizon=5  # giorni
)

print(f"\nLiquidity Risk Assessment:")
print(f"  Time to liquidate 50%: {liquidity_results.time_to_liquidate_50:.1f} days")
print(f"  Market impact cost: {liquidity_results.market_impact_cost:.2%}")
print(f"  Liquidity score: {liquidity_results.liquidity_score:.2f}/10")
```

## Educational

### Adaptive Learning

```python
from src.applications.educational.adaptive_learning import (
    LearningPathOptimizer, StudentModel, ContentRecommender, AssessmentEngine
)

# Modello studente
student_model = StudentModel(
    student_id="student_001",
    learning_style="visual",  # "visual", "auditory", "kinesthetic", "reading"
    knowledge_state={},
    learning_preferences={
        "difficulty_preference": "moderate",
        "pace": "self_paced",
        "feedback_frequency": "immediate"
    }
)

# Ottimizzatore percorso di apprendimento
path_optimizer = LearningPathOptimizer(
    curriculum_graph="math_curriculum.json",
    optimization_objective="mastery_time",  # "mastery_time", "engagement", "retention"
    adaptation_frequency="after_each_activity"
)

# Raccomandatore contenuti
content_recommender = ContentRecommender(
    content_database="educational_content.db",
    recommendation_algorithms=[
        "collaborative_filtering",
        "content_based",
        "knowledge_tracing"
    ],
    personalization_factors=[
        "prior_knowledge", "learning_style", "performance_history"
    ]
)

# Engine di valutazione
assessment_engine = AssessmentEngine(
    question_bank="question_bank.json",
    adaptive_testing=True,
    difficulty_estimation="irt",  # Item Response Theory
    stopping_criteria={
        "max_questions": 20,
        "confidence_threshold": 0.95,
        "time_limit": 1800  # 30 minuti
    }
)

# Inizializza modello studente
await student_model.initialize_knowledge_state(
    subject="mathematics",
    grade_level=8
)

# Valutazione iniziale
initial_assessment = await assessment_engine.conduct_adaptive_assessment(
    student_model,
    subject="algebra",
    assessment_type="diagnostic"
)

print(f"Initial Assessment Results:")
print(f"  Estimated ability: {initial_assessment.ability_estimate:.2f}")
print(f"  Confidence interval: [{initial_assessment.confidence_lower:.2f}, {initial_assessment.confidence_upper:.2f}]")
print(f"  Questions answered: {initial_assessment.questions_answered}")
print(f"  Accuracy: {initial_assessment.accuracy:.1%}")

# Aggiorna modello studente
await student_model.update_knowledge_state(
    assessment_results=initial_assessment,
    subject="algebra"
)

# Ottimizza percorso di apprendimento
learning_path = await path_optimizer.optimize_learning_path(
    student_model=student_model,
    target_concepts=["linear_equations", "quadratic_equations", "systems_of_equations"],
    time_constraint=30  # giorni
)

print(f"\nOptimized Learning Path:")
for i, activity in enumerate(learning_path.activities):
    print(f"  {i+1}. {activity.title} ({activity.type})")
    print(f"     Estimated time: {activity.estimated_time} minutes")
    print(f"     Difficulty: {activity.difficulty_level}")
    print(f"     Prerequisites: {', '.join(activity.prerequisites)}")

# Simula sessione di apprendimento
for activity in learning_path.activities[:5]:  # Prime 5 attività
    print(f"\nStarting activity: {activity.title}")
    
    # Raccomanda contenuti
    recommended_content = await content_recommender.recommend_content(
        student_model=student_model,
        activity=activity,
        num_recommendations=3
    )
    
    print(f"Recommended content:")
    for content in recommended_content:
        print(f"  - {content.title} (relevance: {content.relevance_score:.2f})")
    
    # Simula interazione studente
    interaction_data = {
        "time_spent": random.randint(300, 1200),  # 5-20 minuti
        "completion_rate": random.uniform(0.7, 1.0),
        "correct_answers": random.randint(6, 10),
        "total_questions": 10,
        "help_requests": random.randint(0, 3)
    }
    
    # Aggiorna modello studente
    await student_model.update_from_interaction(
        activity=activity,
        interaction_data=interaction_data
    )
    
    # Valutazione formativa
    if activity.requires_assessment:
        formative_assessment = await assessment_engine.conduct_formative_assessment(
            student_model=student_model,
            concept=activity.target_concept
        )
        
        print(f"Formative assessment: {formative_assessment.mastery_level:.1%} mastery")
        
        # Adatta percorso se necessario
        if formative_assessment.mastery_level < 0.8:
            print("Mastery threshold not met - adapting learning path")
            
            adapted_path = await path_optimizer.adapt_learning_path(
                current_path=learning_path,
                student_model=student_model,
                assessment_results=formative_assessment
            )
            
            learning_path = adapted_path
            print(f"Path adapted: added {len(adapted_path.added_activities)} remedial activities")

# Analisi progresso studente
progress_analysis = await student_model.analyze_learning_progress(
    time_period="last_week"
)

print(f"\nLearning Progress Analysis:")
print(f"  Concepts mastered: {progress_analysis.concepts_mastered}")
print(f"  Average mastery level: {progress_analysis.avg_mastery_level:.1%}")
print(f"  Learning velocity: {progress_analysis.learning_velocity:.2f} concepts/day")
print(f"  Engagement score: {progress_analysis.engagement_score:.2f}/10")
print(f"  Predicted completion time: {progress_analysis.predicted_completion_days} days")
```

### Intelligent Tutoring

```python
from src.applications.educational.intelligent_tutoring import (
    VirtualTutor, DialogueManager, HintGenerator, MistakeAnalyzer
)

# Tutor virtuale
virtual_tutor = VirtualTutor(
    subject_expertise=["mathematics", "physics", "chemistry"],
    tutoring_style="socratic",  # "socratic", "direct", "collaborative"
    personality_traits={
        "patience": 0.9,
        "encouragement": 0.8,
        "formality": 0.6
    },
    language="italian"
)

# Gestore dialogo
dialogue_manager = DialogueManager(
    conversation_model="transformer",
    context_window=10,  # Ultimi 10 scambi
    intent_recognition=True,
    emotion_detection=True
)

# Generatore suggerimenti
hint_generator = HintGenerator(
    hint_strategies=[
        "conceptual_hint",
        "procedural_hint",
        "worked_example",
        "analogy"
    ],
    progressive_disclosure=True,
    personalization=True
)

# Analizzatore errori
mistake_analyzer = MistakeAnalyzer(
    error_taxonomy="mathematics_errors.json",
    misconception_database="common_misconceptions.db",
    remediation_strategies="remediation_rules.json"
)

# Inizia sessione di tutoring
tutoring_session = await virtual_tutor.start_session(
    student_model=student_model,
    topic="solving_quadratic_equations",
    session_goal="master_factoring_method"
)

print(f"Tutoring session started: {tutoring_session.session_id}")
print(f"Tutor: {virtual_tutor.generate_greeting(student_model.name)}")

# Simula conversazione di tutoring
conversation_history = []
student_responses = [
    "Non capisco come fattorizzare x² + 5x + 6",
    "Devo trovare due numeri che moltiplicati danno 6?",
    "2 e 3?",
    "Quindi è (x + 2)(x + 3)?",
    "Come faccio a verificare se è giusto?"
]

for student_input in student_responses:
    print(f"\nStudente: {student_input}")
    
    # Analizza input studente
    input_analysis = await dialogue_manager.analyze_student_input(
        student_input,
        conversation_history,
        current_problem="x² + 5x + 6 = 0"
    )
    
    print(f"Intent detected: {input_analysis.intent}")
    print(f"Emotion: {input_analysis.emotion}")
    print(f"Confidence: {input_analysis.confidence:.2f}")
    
    # Genera risposta del tutor
    if input_analysis.contains_error:
        # Analizza errore
        error_analysis = await mistake_analyzer.analyze_mistake(
            student_input,
            correct_solution="(x + 2)(x + 3)",
            problem_context="factoring_quadratic"
        )
        
        print(f"Error type: {error_analysis.error_type}")
        print(f"Misconception: {error_analysis.misconception}")
        
        # Genera correzione
        tutor_response = await virtual_tutor.generate_error_correction(
            error_analysis,
            student_model,
            tutoring_style="gentle_guidance"
        )
    
    elif input_analysis.needs_hint:
        # Genera suggerimento
        hint = await hint_generator.generate_hint(
            problem="x² + 5x + 6 = 0",
            student_attempt=student_input,
            hint_level=input_analysis.hint_level,
            student_model=student_model
        )
        
        tutor_response = await virtual_tutor.deliver_hint(
            hint,
            encouragement_level=0.8
        )
    
    else:
        # Risposta normale
        tutor_response = await virtual_tutor.generate_response(
            student_input,
            conversation_history,
            session_context=tutoring_session
        )
    
    print(f"Tutor: {tutor_response.text}")
    
    # Aggiorna cronologia conversazione
    conversation_history.append({
        "speaker": "student",
        "text": student_input,
        "timestamp": time.time(),
        "analysis": input_analysis
    })
    
    conversation_history.append({
        "speaker": "tutor",
        "text": tutor_response.text,
        "timestamp": time.time(),
        "response_type": tutor_response.response_type
    })
    
    # Aggiorna modello studente
    await student_model.update_from_tutoring_interaction(
        student_input=student_input,
        tutor_response=tutor_response,
        learning_outcome=input_analysis.learning_progress
    )

# Valutazione sessione
session_evaluation = await virtual_tutor.evaluate_session(
    tutoring_session,
    conversation_history,
    learning_objectives=["understand_factoring", "apply_factoring_method"]
)

print(f"\nSession Evaluation:")
print(f"  Learning objectives achieved: {session_evaluation.objectives_achieved}/{len(session_evaluation.learning_objectives)}")
print(f"  Student engagement: {session_evaluation.engagement_score:.2f}/10")
print(f"  Tutor effectiveness: {session_evaluation.tutor_effectiveness:.2f}/10")
print(f"  Recommended follow-up: {session_evaluation.follow_up_recommendation}")

# Genera report per insegnante
teacher_report = await virtual_tutor.generate_teacher_report(
    student_model=student_model,
    tutoring_session=tutoring_session,
    time_period="this_session"
)

print(f"\nTeacher Report Generated:")
print(f"  Student progress: {teacher_report.progress_summary}")
print(f"  Areas of strength: {', '.join(teacher_report.strengths)}")
print(f"  Areas needing attention: {', '.join(teacher_report.areas_for_improvement)}")
print(f"  Recommended interventions: {', '.join(teacher_report.recommended_interventions)}")
```

## Gestione Errori

```python
# Gestione errori comuni
try:
    result = await robot_controller.execute_trajectory(trajectory)
except RobotControllerError as e:
    logger.error(f"Robot control error: {e}")
    await robot_controller.emergency_stop()
except TrajectoryPlanningError as e:
    logger.error(f"Trajectory planning failed: {e}")
    # Riprova con parametri più conservativi
    
try:
    diagnosis = await diagnostic_ai.diagnose(medical_image)
except ModelLoadError as e:
    logger.error(f"Failed to load diagnostic model: {e}")
    # Usa modello di fallback
except InsufficientImageQualityError as e:
    logger.warning(f"Image quality insufficient: {e}")
    # Richiedi nuova acquisizione

try:
    trade_result = await execute_trade(symbol, action, quantity)
except InsufficientFundsError as e:
    logger.error(f"Insufficient funds for trade: {e}")
except MarketClosedError as e:
    logger.info(f"Market closed, queuing trade: {e}")
    await queue_trade_for_market_open(symbol, action, quantity)
```

## Performance e Ottimizzazione

```python
# Ottimizzazioni per applicazioni real-time

# Robotics - Controllo real-time
robot_controller.set_control_frequency(1000)  # 1kHz
robot_controller.enable_real_time_scheduling()

# Edge AI - Ottimizzazione latenza
edge_inference.enable_tensorrt_optimization()
edge_inference.set_batch_size(1)
edge_inference.enable_fp16_precision()

# Financial - Elaborazione ad alta frequenza
market_feed.enable_low_latency_mode()
trading_strategy.set_execution_priority("high")

# Healthcare - Elaborazione immagini ottimizzata
medical_processor.enable_gpu_acceleration()
medical_processor.set_memory_optimization("aggressive")

# Educational - Risposta interattiva
virtual_tutor.set_response_timeout(2.0)  # 2 secondi max
dialogue_manager.enable_caching()
```