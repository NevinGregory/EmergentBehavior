use avian2d::prelude::*;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use rand::Rng;
use rand::prelude::IndexedRandom;
use std::collections::{HashMap, HashSet};
use std::f32::consts::FRAC_PI_3;

// Neural Network Stuffs
const NN_INPUT_COUNT: usize = 21;
const NN_OUTPUT_COUNT: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq)]
enum NodeType {
    Input,
    Hidden,
    Output,
}

#[derive(Debug, Clone)]
struct Connection {
    from_idx: usize,
    to_idx: usize,
    weight: f32,
    enabled: bool,
    innovation: usize,
}

impl Connection {
    fn new(from_idx: usize, to_idx: usize, weight: f32, innovation: usize) -> Self {
        Self {
            from_idx,
            to_idx,
            weight,
            enabled: true,
            innovation,
        }
    }
}

#[derive(Component, Debug, Clone, Default)]
struct Genome {
    nodes: HashMap<usize, NodeType>,
    connections: Vec<Connection>,
}

#[derive(Component)]
struct NeuralNetwork {
    nodes: Vec<NodeState>,
    execution_order: Vec<usize>,
    output_indices: Vec<usize>,
}

struct NodeState {
    id: usize,
    value: f32,
    incoming: Vec<(usize, f32)>, // (index_in_nodes_vec, weight)
    node_type: NodeType,
}

#[derive(Component)]
struct Fitness(f64);

#[derive(Component)]
struct Generation(u32);

#[derive(Component, Default)]
struct AIOutput {
    direction_x: f32,
    direction_y: f32,
}

#[derive(Component)]
struct UpdateTimer {
    vision_frame: u32,
    ai_frame: u32,
}

impl Default for UpdateTimer {
    fn default() -> Self {
        let mut rng = rand::rng();
        Self {
            // Stagger updates across agents to spread load
            vision_frame: rng.random_range(0..VISION_UPDATE_INTERVAL),
            ai_frame: rng.random_range(0..AI_UPDATE_INTERVAL),
        }
    }
}

#[derive(Resource, Default)]
pub struct InnovationHistory {
    pub map: HashMap<(usize, usize), usize>,      // (from, to) -> innovation_id
    pub node_map: HashMap<usize, usize>,          // connection_id_split -> new_node_id
    pub next_innovation: usize,
    pub next_node_id: usize,
}

#[derive(Resource, Default)]
struct FrameCount(u32);

impl InnovationHistory {
    pub fn new(input_count: usize, output_count: usize) -> Self {
        Self {
            map: HashMap::new(),
            node_map: HashMap::new(),
            // Innovations start at 0, or at (input * output) if fully connected
            next_innovation: input_count * output_count, 
            // Nodes start after all Inputs and Outputs
            next_node_id: input_count + output_count, 
        }
    }
    // For Connections
    pub fn get_innovation(&mut self, from: usize, to: usize) -> usize {
        if let Some(&id) = self.map.get(&(from, to)) {
            id
        } else {
            let id = self.next_innovation;
            self.map.insert((from, to), id);
            self.next_innovation += 1;
            id
        }
    }

    // For Nodes
    pub fn get_node_id(&mut self, split_conn_id: usize) -> usize {
        if let Some(&id) = self.node_map.get(&split_conn_id) {
            id
        } else {
            let id = self.next_node_id;
            self.node_map.insert(split_conn_id, id);
            self.next_node_id += 1;
            id
        }
    }
}

impl Genome  {
    pub fn new_initial(inputs: usize, outputs: usize, history: &mut InnovationHistory) -> Self {
        let mut nodes = HashMap::new();
        let mut connections = Vec::new();

        // 1. Create Input Nodes (IDs: 0 to inputs-1)
        for i in 0..inputs {
            nodes.insert(i, NodeType::Input);
        }

        // 2. Create Output Nodes (IDs: inputs to inputs + outputs - 1)
        for i in 0..outputs {
            nodes.insert(inputs + i, NodeType::Output);
        }

        // 3. Create Initial Connections (Connect every Input to every Output)
        for i in 0..inputs {
            for j in 0..outputs {
                let to_node = inputs + j;
                let innov = history.get_innovation(i, to_node);
                
                connections.push(Connection::new(
                    from_idx: i,
                    to_idx: to_node,
                    weight: rand::random_range(-1.0..1.0),
                    innovation: innov,
                ));
            }
        }

        // 4. Update the InnovationHistory counters
        // Ensure the next node created via mutation starts after our I/O nodes
        if history.next_node_id < (inputs + outputs) {
            history.next_node_id = inputs + outputs;
        }

        Self {
            nodes,
            connections,
        }
    }

    pub fn crossover(parent_a: &Genome, parent_b: &Genome, fitness_a: &Fitness, fitness_b: &Fitness) -> Self {
        let mut child = Genome::default();

        // 1. Determine the fitter parent (needed for disjoint/excess genes)
        let (fitter, other) = if fitness_a.0 > fitness_b.0 {
            (parent_a, parent_b)
        } else if fitness_b.0 > fitness_a.0 {
            (parent_b, parent_a)
        } else {
            // If equal, treat parent_a as fitter but we'll take shared genes randomly
            (parent_a, parent_b)
        };

        // 2. Inherit Nodes
        // In simple NEAT, we take all nodes from both parents or just the fitter.
        // Taking from both ensures the connections have valid IDs.
        child.nodes = fitter.nodes.clone();
        for (id, node_type) in &other.nodes {
            child.nodes.entry(*id).or_insert(*node_type);
        }

        // 3. Inherit Connections
        let other_conns: HashMap<usize, &Connection> = other.connections.iter()
            .map(|c| (c.innovation, c)).collect();

        for conn_f in &fitter.connections {
            if let Some(conn_o) = other_conns.get(&conn_f.innovation) {
                // MATCHING GENE: Pick randomly from either parent
                if rand::random::<bool>() {
                    child.connections.push(conn_f.clone());
                } else {
                    child.connections.push((*conn_o).clone());
                }
            } else {
                // DISJOINT/EXCESS GENE: Inherit from the fitter parent
                child.connections.push(conn_f.clone());
            }
        }

        child
    }

    pub fn compile(&self) -> NeuralNetwork {
        let mut nodes_vec = Vec::new();
        let mut id_to_idx = HashMap::new();

        // 1. DANGER: HashMap iteration is random. 
        // We MUST sort the IDs so Input #1 is always Input #1.
        let mut sorted_ids: Vec<_> = self.nodes.keys().cloned().collect();
        sorted_ids.sort();

        // 2. Map IDs to Vector Indices
        for (idx, id) in sorted_ids.iter().enumerate() {
            id_to_idx.insert(*id, idx);
            nodes_vec.push(NodeState {
                id: *id,
                value: 0.0,
                incoming: Vec::new(),
                node_type: self.nodes[id],
            });
        }

        // 3. Fill the 'incoming' connections for each node
        for conn in self.connections.iter().filter(|c| c.enabled) {
            let to_idx = id_to_idx[&conn.to_idx];
            let from_idx = id_to_idx[&conn.from_idx];
            nodes_vec[to_idx].incoming.push((from_idx, conn.weight));
        }

        // 4. TOPOLOGICAL SORT (Execution Order)
        // This ensures nodes are calculated in the correct order (Inputs -> Hidden -> Outputs)
        let mut execution_order = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = HashSet::new(); // For cycle detection

        fn visit(
            idx: usize,
            nodes: &[NodeState],
            visited: &mut HashSet<usize>,
            stack: &mut HashSet<usize>,
            order: &mut Vec<usize>,
        ) {
            if visited.contains(&idx) || nodes[idx].node_type == NodeType::Input {
                return;
            }
            
            // Cycle detection (NEAT usually prevents this, but safety first)
            if stack.contains(&idx) { return; } 

            stack.insert(idx);
            for (from_idx, _) in &nodes[idx].incoming {
                visit(*from_idx, nodes, visited, stack, order);
            }
            stack.remove(&idx);
            
            visited.insert(idx);
            order.push(idx);
        }

        let output_indices: Vec<usize> = nodes_vec
            .iter()
            .enumerate()
            .filter(|(_, n)| n.node_type == NodeType::Output)
            .map(|(i, _)| i)
            .collect();

        // Start from outputs and work backwards to find all necessary calculations
        for &out_idx in &output_indices {
            visit(out_idx, &nodes_vec, &mut visited, &mut stack, &mut execution_order);
        }

        NeuralNetwork {
            nodes: nodes_vec,
            execution_order,
            output_indices,
        }
    }
}

impl NeuralNetwork {
    pub fn activate(&mut self, inputs: &[f32]) -> Vec<f32> {
        let mut input_ptr = 0;
        for node in &mut self.nodes {
            if node.node_type == NodeType::Input {
                node.value = inputs[input_ptr];
                input_ptr += 1;
            }
        }

        for &idx in &self.execution_order {
            let sum: f32 = self.nodes[idx]
                .incoming
                .iter()
                .map(|(from_idx, weight)| self.nodes[*from_idx].value * weight)
                .sum();
            self.nodes[idx].value = sum.tanh(); // Using Tanh for -1 to 1 output
        }

        self.output_indices
            .iter()
            .map(|&i| self.nodes[i].value)
            .collect()
    }
}

// --- 5. MUTATION LOGIC ---

impl Genome {
    pub fn mutate(&mut self, history: &mut InnovationHistory) {
        let mut rng = rand::rng();
        let mutation_type: f32 = rng.random();

        if mutation_type < 0.8 {
            // 80% Weight Mutation
            for conn in &mut self.connections {
                if rng.random_bool(0.9) {
                    conn.weight += rng.random_range(-0.1..0.1); // Nudge
                } else {
                    conn.weight = rng.random_range(-1.0..1.0); // Reset
                }
            }
        } else if mutation_type < 0.85 {
            // 5% Add Connection
            let keys: Vec<usize> = self.nodes.keys().cloned().collect();
            let from_idx = *keys.choose(&mut rng).unwrap();
            let to_idx = *keys.choose(&mut rng).unwrap();

            // Basic check: don't connect to an input, and don't connect to self
            if self.nodes[&to_idx] != NodeType::Input && from_idx != to_idx {
                let innov = history.get_innovation(from_idx, to_idx);
                self.connections.push(Connection::new(
                    from_idx,
                    to_idx,
                    weight: rng.random_range(-1.0..1.0),
                    innovation: innov,
                ));
            }
        } else if mutation_type < 0.88 {
            let mut new_connections = Vec::new();
            // --- 3% Add Node ---
            if let Some(conn_idx) = self.connections.iter()
                .enumerate()
                .filter(|(_, c)| c.enabled)
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
                .choose(&mut rng) 
            {
                // 1. Disable the chosen connection
                let (from_idx, to_idx, old_weight, old_innov) = {
                    let conn = &mut self.connections[*conn_idx];
                    conn.enabled = false;
                    (conn.from_idx, conn.to_idx, conn.weight, conn.innovation)
                };

                // 2. Get or Create a Node ID for this specific split
                // This ensures structural alignment across the whole population
                let new_node_id = history.get_node_id(old_innov); 

                // 3. Add the hidden node (if it doesn't already exist in this genome)
                self.nodes.insert(new_node_id, NodeType::Hidden);

                // 4. Create two new connections
                let innov1 = history.get_innovation(from_idx, new_node_id);
                let innov2 = history.get_innovation(new_node_id, to_idx);

                new_connections.push(Connection::new(
                    from_idx,
                    to_idx: new_node_id,
                    weight: 1.0, // Preserve signal
                    innovation: innov1,
                ));

                new_connections.push(Connection::new(
                    from_idx: new_node_id,
                    to_idx,
                    weight: old_weight, // Preserve signal
                    innovation: innov2,
                ));
            }
            self.connections.append(&mut new_connections);
        }
    }
}

// Simulation Stuffs

const MAP_SIZE: f32 = 1500.;
const TARGET_SPEED: f32 = 300.;
// How quickly should the camera snap to the desired location.
const CAMERA_DECAY_RATE: f32 = 5.;
const ZOOM_SPEED: f32 = 0.1;
const MIN_ZOOM: f32 = 0.3;
const MAX_ZOOM: f32 = 3.0;
const PLAYER_SCALE: f32 = 64.;
const PLANT_SCALE: f32 = 32.;
const HUNGER_RATE: f32 = 1.0; // How much hunger decreases every gamestep
const HEALING_RATE: f32 = 0.2;
const ENERGY_RATE: f32 = 0.2;
const INITIAL_SPAWN: i32 = 50;
const MAX_POPULATION: usize = 150; // Cap population to prevent slowdown
const COLLISION_DISTANCE: f32 = 20.;
const START_HEALING_TIME: f32 = 10.;
const START_RESTING_TIME: f32 = 2.;
const LIFE_EXPECTANCY: f32 = 70.;

const PREGNANCY_TIME: f32 = 20.;
const REPRODUCTION_AGE: f32 = 18.0;

const VISION_DISTANCE: f32 = 100.0;
const VISION_FOV: f32 = 2.0 * FRAC_PI_3; // 120 degree fov
const MOVEMENT_PER_TICK: f32 = 100.0;

// Performance optimizations
const VISION_UPDATE_INTERVAL: u32 = 3; // Update vision every N frames
const AI_UPDATE_INTERVAL: u32 = 2; // Update neural network every N frames
const MAX_VISION_CHECKS: usize = 50; // Max entities to check for vision

const MALE_COLOR: Color = Color::srgb(0., 0., 1.);
const FEMALE_COLOR: Color = Color::srgb(1., 0., 1.);
const HOVER_COLOR: Color = Color::srgb(1., 0., 0.);

#[derive(PartialEq)]
enum KreacherGender {
    Male,
    Female,
}

#[derive(Component)]
struct Target;

#[derive(Component)]
struct VisionTarget;

#[derive(Component)]
struct Name {
    name: String,
}

#[derive(Component)]
struct Kreacher {
    age: f32,
    gender: KreacherGender,
}

#[derive(Component)]
struct Hunger {
    max_hunger: f32,
    hunger: f32,
}

#[derive(Component)]
struct Health {
    health: f32,
    max_health: f32,
    alive: bool,
    timer: Timer,
}

#[derive(Component)]
struct Energy {
    energy: f32,
    max_energy: f32,
    timer: Timer,
}

#[derive(Component)]
struct Plant;

#[derive(Component)]
struct Edible {
    nutrition_value: f32,
}

#[derive(Component)]
struct Movement {
    velocity: Vec2,
    last_x: f32,
    last_y: f32,
}

#[derive(Component)]
struct Reproducing {
    pregnant: bool,
    timer: Timer,
    child_health: f32,
    child_hunger: f32,
    child_energy: f32,
    child_genome: Option<Genome>,
    is_male: bool,
}

#[derive(Component)]
struct Vision {
    heading: Vec2,
    seeing: Vec<String>,
    closest_food: Option<Vec2>,
    closest_predator: Option<Vec2>,
    closest_mate: Option<Vec2>,
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, PhysicsPlugins::default()))
        .add_plugins(MeshPickingPlugin)
        .add_systems(Startup, (setup_scene, setup_camera, setup_ui))
        .add_systems(
            Update,
            (
                increment_frame_count,
                ((update_health, update_hunger, update_energy), despawn_dead).chain(),
                (move_target, update_camera).chain(),
                camera_zoom,
                kreacher_eating_collision,
                kreacher_reproducing_collision,
                update_ui,
                update_heading,
                update_reproduction,
                update_age,
                (update_vision, agent_sensory_system, move_kreacher).chain(),
                update_fitness,
                cull_population,
            ),
        )
        .insert_resource(InnovationHistory::new(NN_INPUT_COUNT, NN_OUTPUT_COUNT))
        .insert_resource(FrameCount(0))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
    mut history: ResMut<InnovationHistory>,
) {
    // World where we move the target
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(MAP_SIZE, MAP_SIZE))),
        MeshMaterial2d(materials.add(Color::srgb(0.2, 0.2, -1.))),
    ));

    // Target
    commands
        .spawn((
            Target,
            Transform::from_xyz(0., 0., 0.),
            /*
            Kreacher {
                age: 10.,
                gender: KreacherGender::Male,
            },
            Sprite {
                image: asset_server.load("human.png"),
                color: Color::srgb(0., 0., 1.),
                custom_size: Some(Vec2::new(PLAYER_SCALE, PLAYER_SCALE)),
                ..default()
            },
            Pickable {
                should_block_lower: true,
                is_hoverable: true,
            },
            Hunger {
                hunger: 100.,
                max_hunger: 100.,
            },
            Health {
                health: 100.,
                max_health: 100.,
                alive: true,
                timer: Timer::from_seconds(START_HEALING_TIME, TimerMode::Once),
            },
            Energy {
                energy: 100.,
                max_energy: 100.,
                timer: Timer::from_seconds(START_RESTING_TIME, TimerMode::Once),
            },
            Movement {
                velocity: Vec2::new(0.0, 0.0),
                last_x: 0.0,
                last_y: 0.0,
            },
            Collider::circle(5.0),
            CollisionEventsEnabled,
            CollidingEntities::default(),
            Vision {
                heading: Vec2::new(1.0, 0.0),
                seeing: Vec::new(),
                closest_food: None,
                closest_predator: None,
                closest_mate: None,
            },
            */
            Name {
                name: "Player".to_string(),
            },
            //VisionTarget,
        ))
        .insert(Sensor);

    let mut rng = rand::rng();
    let base_genome = Genome::new_initial(NN_INPUT_COUNT, NN_OUTPUT_COUNT, &mut history);
    let spawn_range = MAP_SIZE / 2.0 * 0.8; // Spawn within 80% of map bounds
    for _ in 1..=INITIAL_SPAWN {
        let x: f32 = rng.random_range(-spawn_range..=spawn_range);
        let y: f32 = rng.random_range(-spawn_range..=spawn_range);

        let max_hunger: f32 = rng.random_range(50.0..=200.0);
        let max_health: f32 = rng.random_range(50.0..=200.0);
        let max_energy: f32 = rng.random_range(100.0..=150.0);

        let is_male: bool = rng.random_bool(0.5);

        // Create diverse initial population with multiple mutations
        let mut kreacher_genome = base_genome.clone();
        for _ in 0..5 {
            kreacher_genome.mutate(&mut history);
        }
        let nn = kreacher_genome.compile();

        // Vary starting age to prevent synchronized reproduction
        let start_age: f32 = rng.random_range(0.0..=15.0);

        //Kreacher
        spawn_kreacher(
            &mut commands,
            &asset_server,
            Vec2::new(x, y),
            is_male,
            max_hunger,
            max_health,
            max_energy,
            start_age,
            kreacher_genome,
            nn,
            Generation(0)
        );
    }

    //Plant
    for _ in 1..=INITIAL_SPAWN {
        let x: f32 = rng.random_range(-spawn_range..=spawn_range);
        let y: f32 = rng.random_range(-spawn_range..=spawn_range);

        commands.spawn((
            Plant,
            Health {
                health: 100.,
                max_health: 100.,
                alive: true,
                timer: Timer::from_seconds(START_HEALING_TIME, TimerMode::Once),
            },
            Sprite {
                image: asset_server.load("plant.png"),
                color: Color::srgb(0., 1., 0.),
                custom_size: Some(Vec2::new(PLANT_SCALE, PLANT_SCALE)),
                ..default()
            },
            Edible {
                nutrition_value: 100.,
            },
            VisionTarget,
            Name {
                name: "Plant".to_string(),
            },
            Collider::circle(5.0),
            Transform::from_xyz(x, y, 0.),
        ));
    }
}

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Population: "),
        Node {
            position_type: PositionType::Absolute,
            bottom: px(12),
            left: px(12),
            ..default()
        },
    ));
}

fn update_ui(
    mut text_query: Query<&mut Text>,
    //target_query: Single<(&Hunger, &Health, &Energy, &Movement, &Vision), With<Target>>,
    kreacher_query: Query<&Kreacher>,
) {
    let population = kreacher_query.iter().count();
    
    for mut text in text_query.iter_mut() {
        *text = Text::new(format!(
            "Population: {}"/*\nHunger {:.2} Health {:.2} Energy {:.2}\nVelocity ({:.2}, {:.2}) Seeing: [{:?}]"*/,
            population,
            //target_query.0.hunger,
            //target_query.1.health,
            //target_query.2.energy,
            //target_query.3.velocity.x,
            //target_query.3.velocity.y,
            //target_query.4.seeing,
        ));
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// Update the camera position by tracking the target.
fn update_camera(
    mut camera: Single<&mut Transform, (With<Camera2d>, Without<Target>)>,
    target: Single<&Transform, (With<Target>, Without<Camera2d>)>,
    time: Res<Time>,
) {
    let Vec3 { x, y, .. } = target.translation;
    let direction = Vec3::new(x, y, camera.translation.z);

    // Applies a smooth effect to camera movement using stable interpolation
    // between the camera position and the target position on the x and y axes.
    camera
        .translation
        .smooth_nudge(&direction, CAMERA_DECAY_RATE, time.delta_secs());
}

/// Handle camera zoom with scroll wheel.
fn camera_zoom(
    mut scroll_events: MessageReader<MouseWheel>,
    mut camera_query: Query<&mut Projection, With<Camera2d>>,
) {
    for event in scroll_events.read() {
        if let Ok(mut projection) = camera_query.single_mut() {
            if let Projection::Orthographic(ortho) = projection.as_mut() {
                // Scroll up = zoom in (decrease scale), scroll down = zoom out (increase scale)
                let zoom_delta = match event.unit {
                    MouseScrollUnit::Line => -event.y * ZOOM_SPEED,
                    MouseScrollUnit::Pixel => -event.y * ZOOM_SPEED * 0.01,
                };

                ortho.scale = (ortho.scale + zoom_delta).clamp(MIN_ZOOM, MAX_ZOOM);
            }
        }
    }
}

/// Update the target position with keyboard inputs.
fn move_target(
    mut target: Single<&mut Transform, With<Target>>,
    time: Res<Time>,
    kb_input: Res<ButtonInput<KeyCode>>,
) {
    let mut direction = Vec2::ZERO;

    if kb_input.pressed(KeyCode::KeyW) {
        direction.y += 1.;
    }

    if kb_input.pressed(KeyCode::KeyS) {
        direction.y -= 1.;
    }

    if kb_input.pressed(KeyCode::KeyA) {
        direction.x -= 1.;
    }

    if kb_input.pressed(KeyCode::KeyD) {
        direction.x += 1.;
    }

    let move_delta = direction.normalize_or_zero() * TARGET_SPEED * time.delta_secs();
    target.translation += move_delta.extend(0.);
}

fn update_hunger(time: Res<Time>, mut hunger_query: Query<&mut Hunger>) {
    hunger_query.iter_mut().for_each(|mut query| {
        query.hunger -= HUNGER_RATE * time.delta_secs();

        if query.hunger <= 0. {
            query.hunger = 0.;
        }
    });
}

fn update_energy(time: Res<Time>, mut targets: Query<(&mut Energy, &Movement), With<Kreacher>>) {
    targets.iter_mut().for_each(|(mut energy, movement)| {
        energy.timer.tick(time.delta());

        // If moving, lose energy? Proportional to speed?
        if movement.velocity.abs().length() > f32::EPSILON {
            energy.energy -= ENERGY_RATE * time.delta_secs();
            energy.timer.reset();
        }

        if energy.timer.is_finished() && energy.energy < energy.max_energy {
            energy.energy += ENERGY_RATE * time.delta_secs();
        }
    });
}

fn update_health(time: Res<Time>, mut targets: Query<(&mut Health, &Hunger), With<Hunger>>) {
    targets.iter_mut().for_each(|(mut health, hunger)| {
        health.timer.tick(time.delta());

        if hunger.hunger <= 0. {
            health.health -= HUNGER_RATE * time.delta_secs();
            health.timer.reset();
        }

        if health.health <= 0. {
            health.health = 0.;
            health.alive = false;
        }

        if health.timer.is_finished() && health.health < health.max_health {
            health.health += HEALING_RATE * time.delta_secs();
        }
    });
}

fn update_age(time: Res<Time>, mut targets: Query<(&mut Kreacher, &mut Health)>) {
    targets.iter_mut().for_each(|(mut kreacher, mut health)| {
        if health.alive {
            kreacher.age += time.delta_secs();

            if kreacher.age > LIFE_EXPECTANCY {
                health.alive = false;
            }
        }
    });
}

fn update_heading(time: Res<Time>, mut targets: Query<(&Transform, &mut Movement, &mut Vision)>) {
    targets
        .iter_mut()
        .for_each(|(transform, mut movement, mut vision)| {
            let delta_x = transform.translation.x - movement.last_x;
            let delta_y = transform.translation.y - movement.last_y;
            movement.velocity.x = delta_x / time.delta_secs();
            movement.velocity.y = delta_y / time.delta_secs();

            if movement.velocity != Vec2::ZERO {
                vision.heading.x = movement.velocity.normalize_or_zero().x;
                vision.heading.y = movement.velocity.normalize_or_zero().y;
            }

            movement.last_x = transform.translation.x;
            movement.last_y = transform.translation.y;
        });
}

fn update_reproduction(
    time: Res<Time>,
    mut targets: Query<(&Transform, &mut Reproducing, &Generation)>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    targets.iter_mut().for_each(|(transform, mut reproducing, generation)| {
        if reproducing.pregnant {
            reproducing.timer.tick(time.delta());
            if reproducing.timer.is_finished() {
                if let Some(ref genome) = reproducing.child_genome {
                    let nn = genome.compile();
                    //println!("A Kreacher has spawned");
                    spawn_kreacher(
                        &mut commands,
                        &asset_server,
                        Vec2::new(transform.translation.x, transform.translation.y),
                        reproducing.is_male,
                        reproducing.child_hunger,
                        reproducing.child_health,
                        reproducing.child_energy,
                        0.0,
                        genome.clone(),
                        nn,
                        Generation(generation.0 + 1),
                    );

                    reproducing.pregnant = false;
                }
            }
        }
    });
}

fn despawn_dead(mut commands: Commands, query: Query<(Entity, &Health), With<Health>>) {
    query.iter().for_each(|(entity, entity_health)| {
        if !entity_health.alive {
            commands.entity(entity).despawn();
        }
    });
}

fn kreacher_eating_collision(
    mut collision_event_reader: MessageReader<CollisionStart>,
    mut edible_collider_query: Query<(&mut Transform, &Edible)>,
    mut kreacher_collider_query: Query<(&mut Hunger, &mut Fitness), With<Kreacher>>,
) {
    for CollisionStart {
        collider1: e1,
        collider2: e2,
        ..
    } in collision_event_reader.read()
    {
        let collision_pair = if let Ok((hunger, fitness)) = kreacher_collider_query.get_mut(*e1) {
            edible_collider_query
                .get_mut(*e2)
                .map(|edible| (hunger, fitness, edible))
                .ok()
        } else if let Ok((hunger, fitness)) = kreacher_collider_query.get_mut(*e2) {
            edible_collider_query
                .get_mut(*e1)
                .map(|edible| (hunger, fitness, edible))
                .ok()
        } else {
            None
        };

        if let Some((mut hunger, mut fitness, (mut edible_transform, edible))) = collision_pair {
            // "Despawn" eaten thing (Move it somewhere else)
            let mut rng = rand::rng();
            // Clamp position to map boundaries
            let half_map = MAP_SIZE / 2.0;
            let x: f32 = rng.random_range(-half_map..=half_map);
            let y: f32 = rng.random_range(-half_map..=half_map);

            edible_transform.translation = Vec3::new(x, y, 0.);

            hunger.hunger += edible.nutrition_value;
            if hunger.hunger > hunger.max_hunger {
                hunger.hunger = hunger.max_hunger;
            }

            // Reward finding and eating food
            fitness.0 += 10.0;
        }
    }
}

fn kreacher_reproducing_collision(
    mut collision_event_reader: MessageReader<CollisionStart>,
    mut kreacher_query: Query<(
        &Transform,
        &Hunger,
        &Health,
        &Energy,
        &Kreacher,
        &Genome,
        &mut Fitness,
        &mut Reproducing,
    )>,
    mut innov_history: ResMut<InnovationHistory>,
) {
    for CollisionStart {
        collider1: e1,
        collider2: e2,
        ..
    } in collision_event_reader.read()
    {
        if let Ok([b1, b2]) = kreacher_query.get_many_mut([*e1, *e2]) {
            let (transform1, hunger1, health1, energy1, kreacher1, genome1, mut fitness1, reproducing1) = b1;
            let (transform2, hunger2, health2, energy2, kreacher2, genome2, mut fitness2, reproducing2) = b2;

            // Require good health, hunger, and energy to reproduce
            let can_reproduce1 = kreacher1.age > REPRODUCTION_AGE
                && hunger1.hunger > hunger1.max_hunger * 0.8
                && health1.health > health1.max_health * 0.5
                && energy1.energy > energy1.max_energy * 0.5;

            let can_reproduce2 = kreacher2.age > REPRODUCTION_AGE
                && hunger2.hunger > hunger2.max_hunger * 0.8
                && health2.health > health2.max_health * 0.5
                && energy2.energy > energy2.max_energy * 0.5;

            if kreacher1.gender != kreacher2.gender
                && !reproducing1.pregnant
                && !reproducing2.pregnant
                && can_reproduce1
                && can_reproduce2
            {
                // One is male, one is female
                let dist = transform1
                    .translation
                    .truncate()
                    .distance(transform2.translation.truncate());
                if dist < COLLISION_DISTANCE {
                    // They're close enough...
                    let mut rng = rand::rng();
                    // Now we combine them
                    let will_reproduce = rng.random_bool(0.6);
                    if will_reproduce {
                        let mut reproducing = if kreacher1.gender == KreacherGender::Female {
                            reproducing1
                        } else {
                            reproducing2
                        };
                        let child_hunger = ((hunger1.max_hunger + hunger2.max_hunger) / 2.)
                            + rng.random_range(-10_f32..10_f32);
                        let child_health = ((health1.max_health + health2.max_health) / 2.)
                            + rng.random_range(-10_f32..10_f32);
                        let child_energy = ((energy1.max_energy + energy2.max_energy) / 2.)
                            + rng.random_range(-10_f32..10_f32);
                        let mut baby_genome = Genome::crossover(genome1, genome2, &fitness1, &fitness2);
                        baby_genome.mutate(&mut innov_history);
                        let is_male: bool = rng.random_bool(0.5);

                        reproducing.pregnant = true;
                        reproducing.child_health = child_health;
                        reproducing.child_hunger = child_hunger;
                        reproducing.child_energy = child_energy;
                        reproducing.child_genome = Some(baby_genome);
                        reproducing.is_male = is_male;

                        // Reward both parents for successful reproduction
                        fitness1.0 += 100.0;
                        fitness2.0 += 100.0;
                    }
                }
            }
        }
    }
}

fn spawn_kreacher(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    spawn_loc: Vec2,
    is_male: bool,
    hunger: f32,
    health: f32,
    energy: f32,
    age: f32,
    genome: Genome,
    nn: NeuralNetwork,
    generation: Generation,
) {
    let start_color: Color = if is_male { MALE_COLOR } else { FEMALE_COLOR };

    commands
        .spawn((
            Kreacher {
                age: age,
                gender: if is_male {
                    KreacherGender::Male
                } else {
                    KreacherGender::Female
                },
            },
            Hunger {
                hunger: hunger,
                max_hunger: hunger,
            },
            Health {
                health: health,
                max_health: health,
                alive: true,
                timer: Timer::from_seconds(START_HEALING_TIME, TimerMode::Once),
            },
            Sprite {
                image: asset_server.load("human.png"),
                color: start_color,
                custom_size: Some(Vec2::new(PLAYER_SCALE, PLAYER_SCALE)),
                ..default()
            },
            Energy {
                energy: energy,
                max_energy: energy,
                timer: Timer::from_seconds(START_RESTING_TIME, TimerMode::Once),
            },
            Transform::from_xyz(spawn_loc.x, spawn_loc.y, 0.),
            Pickable {
                should_block_lower: true,
                is_hoverable: true,
            },
            Reproducing {
                pregnant: false,
                timer: Timer::from_seconds(PREGNANCY_TIME, TimerMode::Once),
                child_health: 0.0,
                child_hunger: 0.0,
                child_energy: 0.0,
                child_genome: None,
                is_male: false,
            },
            Collider::circle(5.0),
            CollisionEventsEnabled,
            CollidingEntities::default(),
            Vision {
                heading: Vec2::new(1.0, 0.0),
                seeing: Vec::new(),
                closest_food: None,
                closest_predator: None,
                closest_mate: None,
            },
            VisionTarget,
            Name {
                name: "Kreacher".to_string(),
            },
            Movement {
                velocity: Vec2::new(0.0, 0.0),
                last_x: 0.0,
                last_y: 0.0,
            },
        ))
        .insert((
            genome,
            nn,
            generation,
            Fitness(0.0),
            AIOutput::default(),
            UpdateTimer::default(),
        ))
        .insert(Sensor)
        .observe(
            |trigger: On<Pointer<Click>>, query: Query<(&Hunger, &Health, &Generation)>| {
                let clicked_entity = trigger.entity;

                if let Ok((hunger, health, _generation)) = query.get(clicked_entity) {
                    println!("Hunger: {}, Health: {}", hunger.hunger, health.health);
                }
            },
        )
        .observe(
            move |trigger: On<Pointer<Over>>, mut query: Query<&mut Sprite>| {
                if let Ok(mut sprite_handle) = query.get_mut(trigger.entity) {
                    sprite_handle.color = HOVER_COLOR;
                }
            },
        )
        .observe(
            move |trigger: On<Pointer<Out>>, mut query: Query<&mut Sprite>| {
                if let Ok(mut sprite_handle) = query.get_mut(trigger.entity) {
                    sprite_handle.color = start_color;
                }
            },
        );
}

fn move_kreacher(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &AIOutput), (With<Kreacher>, Without<Target>)>,
) {
    let half_map = MAP_SIZE / 2.0;

    query.iter_mut().for_each(|(mut transform, ai_output)| {
        let movement_unit = MOVEMENT_PER_TICK * time.delta_secs();

        // Use neural network outputs directly as movement direction
        let direction = Vec2::new(ai_output.direction_x, ai_output.direction_y).normalize_or_zero();

        transform.translation += Vec3::new(
            direction.x * movement_unit,
            direction.y * movement_unit,
            0.0
        );

        // Clamp position to map boundaries
        transform.translation.x = transform.translation.x.clamp(-half_map, half_map);
        transform.translation.y = transform.translation.y.clamp(-half_map, half_map);
    });
}

// TODO Add raycast for field of vision
fn update_vision(
    spatial_query: SpatialQuery,
    frame_count: Res<FrameCount>,
    mut viewer_query: Query<(Entity, &Transform, &mut Vision, &Kreacher, &mut UpdateTimer)>,
    vision_target_query: Query<(Entity, &Transform, &Name, Option<&Kreacher>), With<VisionTarget>>,
    //mut gizmos: Gizmos,
) {
    viewer_query.iter_mut().for_each(
        |(viewer_ent, viewer_transform, mut viewer_vision, viewer_kreacher, update_timer)| {
            if frame_count.0 % VISION_UPDATE_INTERVAL != update_timer.vision_frame {
                return;
            }
            let viewer_pos = viewer_transform.translation.truncate();
            viewer_vision.seeing.clear();
            viewer_vision.closest_food = None;
            viewer_vision.closest_predator = None;
            viewer_vision.closest_mate = None;

            // Pre-filter and sort by distance for performance
            let mut nearby_targets: Vec<_> = vision_target_query
                .iter()
                .filter_map(|(target_ent, target_transform, target_name, target_kreacher)| {
                    let target_pos = target_transform.translation.truncate();
                    let distance = viewer_pos.distance(target_pos);

                    if distance < VISION_DISTANCE {
                        Some((distance, target_ent, target_pos, target_name, target_kreacher))
                    } else {
                        None
                    }
                })
                .collect();

            // Sort by distance and limit checks
            nearby_targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            nearby_targets.truncate(MAX_VISION_CHECKS);

            for (distance, target_ent, target_pos, target_name, target_kreacher) in nearby_targets {
                let to_target = target_pos - viewer_pos;

                if viewer_vision.heading.angle_to(to_target).abs() < VISION_FOV {
                        let ray_direction = to_target.normalize_or_zero();

                        let filter = SpatialQueryFilter::from_excluded_entities([viewer_ent]);

                        if let Some(hit) = spatial_query.cast_ray(
                            viewer_pos,
                            Dir2::new(ray_direction).unwrap(),
                            distance,
                            true,
                            &filter,
                        ) {
                            if hit.entity == target_ent {
                                let side_vec = Vec2::new(viewer_vision.heading.y, -viewer_vision.heading.x);
                                let local_x = to_target.dot(side_vec);
                                let local_y = to_target.dot(viewer_vision.heading);

                                let local_pos = Vec2::new(local_x, local_y);
                                match target_name.name.as_str() {
                                    "Kreacher" => {
                                        // check if it is a potential mate
                                        if let Some(target_kreacher) = target_kreacher
                                            && target_kreacher.age > REPRODUCTION_AGE
                                            && target_kreacher.gender != viewer_kreacher.gender
                                        {
                                            viewer_vision.seeing.push("Mate".to_string());
                                            if viewer_vision.closest_mate.map_or(true, |d| distance < d.length()) {
                                                viewer_vision.closest_mate = Some(local_pos);
                                            }
                                        }
                                    }
                                    "Plant" => {
                                        viewer_vision.seeing.push("Plant".to_string());
                                        if viewer_vision.closest_food.map_or(true, |d| distance < d.length()) {
                                            viewer_vision.closest_food = Some(local_pos);
                                        }
                                    }
                                    "Predator" => {
                                        viewer_vision.seeing.push("Predator".to_string());
                                        if viewer_vision.closest_predator.map_or(true, |d| distance < d.length()) {
                                            viewer_vision.closest_predator = Some(local_pos);
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                }
            }
        },
    );
}

fn agent_sensory_system(
    frame_count: Res<FrameCount>,
    mut query: Query<(
        &mut NeuralNetwork,
        &mut AIOutput,
        &Hunger,
        &Health,
        &Energy,
        &Kreacher,
        &Movement,
        &Vision,
        &Reproducing,
        &mut UpdateTimer,
    ), Without<Target>>,
) {
    query.iter_mut().for_each(
        |(mut nn, mut ai_output, hunger, health, energy, kreacher, movement, vision, reproducing, update_timer)| {
            // Staggered AI updates: only run NN every N frames
            if frame_count.0 % AI_UPDATE_INTERVAL != update_timer.ai_frame {
                return;
            }
            // --- 1. Assemble the Input Vector ---
            let mut inputs = Vec::with_capacity(NN_INPUT_COUNT);

            inputs.push(hunger.hunger / hunger.max_hunger);
            inputs.push(health.health / health.max_health);
            inputs.push(energy.energy / energy.max_energy);
            inputs.push(kreacher.age / LIFE_EXPECTANCY);

            inputs.push(if kreacher.gender == KreacherGender::Female {
                1.0
            } else {
                0.0
            });
            inputs.push(if reproducing.pregnant { 1.0 } else { 0.0 });

            inputs.push(movement.velocity.length() / 141.42);
            inputs.push(vision.heading.x);
            inputs.push(vision.heading.y);

            // --- 2. Handle the Vision (Strings) ---
            inputs.push(if vision.seeing.contains(&"Plant".to_string()) {
                1.0
            } else {
                0.0
            });
            inputs.push(if vision.seeing.contains(&"Predator".to_string()) {
                1.0
            } else {
                0.0
            });
            inputs.push(if vision.seeing.contains(&"Mate".to_string()) {
                1.0
            } else {
                0.0
            });

            if let Some(pos) = vision.closest_food {
                inputs.push(1.0 - (pos.length() / VISION_DISTANCE).min(1.0));
                inputs.push(pos.x / VISION_DISTANCE);
                inputs.push(pos.y / VISION_DISTANCE);
            } else {
                inputs.extend([0.0, 0.0, 0.0]); // See nothing
            }

            if let Some(pos) = vision.closest_mate {
                inputs.push(1.0 - (pos.length() / VISION_DISTANCE).min(1.0));
                inputs.push(pos.x / VISION_DISTANCE);
                inputs.push(pos.y / VISION_DISTANCE);
            } else {
                inputs.extend([0.0, 0.0, 0.0]); // See nothing
            }

            if let Some(pos) = vision.closest_predator {
                inputs.push(1.0 - (pos.length() / VISION_DISTANCE).min(1.0));
                inputs.push(pos.x / VISION_DISTANCE);
                inputs.push(pos.y / VISION_DISTANCE);
            } else {
                inputs.extend([0.0, 0.0, 0.0]); // See nothing
            }

            // --- 3. Activate the Neural Network ---
            let outputs = nn.activate(&inputs);

            // --- 4. Store outputs for movement system with exploration noise ---
            if outputs.len() >= 2 {
                let mut rng = rand::rng();
                // Add exploration noise to encourage diverse behaviors (decreases with hunger)
                let exploration_factor = 0.3 * (hunger.hunger / hunger.max_hunger);
                let noise_x = rng.random_range(-exploration_factor..=exploration_factor);
                let noise_y = rng.random_range(-exploration_factor..=exploration_factor);

                ai_output.direction_x = outputs[0] + noise_x;
                ai_output.direction_y = outputs[1] + noise_y;
            }
        },
    );
}

fn increment_frame_count(mut frame_count: ResMut<FrameCount>) {
    frame_count.0 = frame_count.0.wrapping_add(1);
}

fn cull_population(
    mut commands: Commands,
    kreacher_query: Query<(Entity, &Fitness), With<Kreacher>>,
) {
    let population = kreacher_query.iter().count();

    if population > MAX_POPULATION {
        let cull_count = population - MAX_POPULATION;

        // Sort by fitness (lowest first) and remove the weakest
        let mut creatures: Vec<_> = kreacher_query.iter().collect();
        creatures.sort_by(|a, b| a.1.0.partial_cmp(&b.1.0).unwrap());

        for (entity, _) in creatures.iter().take(cull_count) {
            commands.entity(*entity).despawn();
        }
    }
}

fn update_fitness(
    time: Res<Time>,
    mut query: Query<(&mut Fitness, &Health), With<Kreacher>>,
) {
    query.iter_mut().for_each(|(mut fitness, health)| {
        if health.alive {
            // Increase fitness for staying alive (1 point per second)
            fitness.0 += time.delta_secs() as f64;
        }
    });
}
