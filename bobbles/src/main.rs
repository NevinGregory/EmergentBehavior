use bevy::{
    post_process::bloom::Bloom, prelude::*
};
use rand::Rng;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

// Neural Network Stuffs
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

#[derive(Debug, Clone, Default)]
struct Genome {
    nodes: HashMap<usize, NodeType>,
    connections: Vec<Connection>,
    pub fitness: f32,
}


#[derive(Resource, Default)]
struct InnovationTracker {
    current_number: usize,
    history: HashMap<(usize, usize), usize>, // (from, to) -> innovation_id
}

#[derive(Component)]
struct NeuralNetwork {
    nodes: Vec<NodeState>,
    execution_order: Vec<usize>,
    inputs_count: usize,
    output_indices: Vec<usize>,
}

pub struct NodeState {
    pub id: usize,
    pub value: f32,
    pub incoming: Vec<(usize, f32)>, // (index_in_nodes_vec, weight)
    pub node_type: NodeType,
}

#[derive(Component)]
struct Fitness(f64);

#[derive(Resource, Default)]
pub struct InnovationHistory {
    pub map: HashMap<(usize, usize), usize>,
    pub next_innovation: usize,
    pub next_node_id: usize,
}

impl InnovationHistory {
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
}

impl Genome {
    pub fn compile(&self) -> NeuralNetwork {
        let mut nodes_vec = Vec::new();
        let mut id_to_idx = HashMap::new();
        
        for (id, node_type) in &self.nodes {
            id_to_idx.insert(*id, nodes_vec.len());
            nodes_vec.push(NodeState {
                id: *id,
                value: 0.0,
                incoming: Vec::new(),
                node_type: *node_type,
            });
        }

        for conn in self.connections.iter().filter(|c| c.enabled) {
            let to_idx = id_to_idx[&conn.to_idx];
            let from_idx = id_to_idx[&conn.from_idx];
            nodes_vec[to_idx].incoming.push((from_idx, conn.weight));
        }

        let mut execution_order = Vec::new();
        let mut visited = HashSet::new();
        
        fn visit(
            idx: usize, 
            nodes: &Vec<NodeState>, 
            visited: &mut HashSet<usize>, 
            order: &mut Vec<usize>,
            id_to_idx: &HashMap<usize, usize>
        ) {
            if visited.contains(&idx) || nodes[idx].node_type == NodeType::Input { return; }
            for (from_idx, _) in &nodes[idx].incoming {
                visit(*from_idx, nodes, visited, order, id_to_idx);
            }
            visited.insert(idx);
            order.push(idx);
        }

        let output_indices: Vec<usize> = nodes_vec.iter().enumerate()
            .filter(|(_, n)| n.node_type == NodeType::Output)
            .map(|(i, _)| i).collect();

        for &out_idx in &output_indices {
            visit(out_idx, &nodes_vec, &mut visited, &mut execution_order, &id_to_idx);
        }

        NeuralNetwork {
            nodes: nodes_vec,
            execution_order,
            inputs_count: self.nodes.values().filter(|&&t| t == NodeType::Input).count(),
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
            let sum: f32 = self.nodes[idx].incoming.iter()
                .map(|(from_idx, weight)| self.nodes[*from_idx].value * weight)
                .sum();
            self.nodes[idx].value = sum.tanh(); // Using Tanh for -1 to 1 output
        }

        self.output_indices.iter().map(|&i| self.nodes[i].value).collect()
    }
}

// --- 5. MUTATION LOGIC ---

impl Genome {
    pub fn mutate(&mut self, history: &mut InnovationHistory) {
        let mut rng = rand::rng();
        let mutation_type: f32 = rng.random();

        if mutation_type < 0.8 { // 80% Weight Mutation
            for conn in &mut self.connections {
                if rng.random_bool(0.9) {
                    conn.weight += rng.random_range(-0.1..0.1); // Nudge
                } else {
                    conn.weight = rng.random_range(-1.0..1.0); // Reset
                }
            }
        } else if mutation_type < 0.85 { // 5% Add Connection
            let keys: Vec<usize> = self.nodes.keys().cloned().collect();
            let from_idx = *keys.choose(&mut rng).unwrap();
            let to_idx = *keys.choose(&mut rng).unwrap();
            
            // Basic check: don't connect to an input, and don't connect to self
            if self.nodes[&to_idx] != NodeType::Input && from_idx != to_idx {
                let innov = history.get_innovation(from_idx, to_idx);
                self.connections.push(Connection {
                    from_idx, to_idx, weight: rng.random_range(-1.0..1.0), enabled: true, innovation: innov,
                });
            }
        } else if mutation_type < 0.88 { // 3% Add Node
            if let Some(conn) = self.connections.iter_mut().filter(|c| c.enabled).choose(&mut rng) {
                conn.enabled = false;
                let new_id = history.next_node_id;
                history.next_node_id += 1;
                
                self.nodes.insert(new_id, NodeType::Hidden);
                
                // Add two connections to replace the old one
                let innov1 = history.get_innovation(conn.from_idx, new_id);
                let innov2 = history.get_innovation(new_id, conn.to_idx);
                
                //self.connections.push(Connection { from_idx: conn.from_idx, to_idx: new_id, weight: 1.0, enabled: true, innovation: innov1 });
                //self.connections.push(Connection { from_idx: new_id, to_idx: conn.to_idx, weight: conn.weight, enabled: true, innovation: innov2 });
            }
        }
    }
}

// Simulation Stuffs

/// Target movement speed factor.
const TARGET_SPEED: f32 = 200.;
/// How quickly should the camera snap to the desired location.
const CAMERA_DECAY_RATE: f32 = 5.;
const PLAYER_SCALE: f32 = 64.;
const PLANT_SCALE: f32 = 32.;
const HUNGER_RATE: f32 = 1.0; // How much hunger decreases every gamestep
const HEALING_RATE: f32 = 0.2; 
const ENERGY_RATE: f32 = 0.2; 
const INITIAL_SPAWN: i32 = 20;
const COLLISION_DISTANCE: f32 = 8.;
const START_HEALING_TIME: f32 = 10.;
const START_RESTING_TIME: f32 = 2.;

const PREGNANCY_TIME: f32 = 20.;

const MALE_COLOR: Color = Color::srgb(0., 0., 1.);
const FEMALE_COLOR: Color = Color::srgb(1., 0., 1.);

enum BobbleGender {
    Male,
    Female
}

#[derive(Component)]
struct Target;

#[derive(Component)]
struct Bobble {
    age: i32,
    gender: BobbleGender,
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
struct Collider;

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
    
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(MeshPickingPlugin) 
        .add_systems(Startup, (setup_scene, setup_camera, setup_ui))
        .add_systems(Update, (
            ((update_health, update_hunger, update_energy), despawn_dead).chain(),
            (move_target, update_camera).chain(),
            bobble_eating_collision,
            update_ui,
            update_velocity,
        ))
        .run();
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
) {
    // World where we move the target
    commands.spawn((
        Mesh2d(meshes.add(Rectangle::new(1000., 1000.))),
        MeshMaterial2d(materials.add(Color::srgb(0.2, 0.2, -1.))),
    ));

    // Target
    commands.spawn((
        Target,
        Transform::from_xyz(0., 0., 0.),
        Bobble {
            age: 10,
            gender: BobbleGender::Male,
        },
        Sprite {
            image: asset_server.load("human.png"),
            color: Color::srgb(1., 0., 1.),
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
        Collider,
    ));

    let mut rng = rand::rng();
    for _ in 1..=INITIAL_SPAWN {
        let x: f32 = rng.random_range(-500_f32..=500_f32);
        let y: f32 = rng.random_range(-500_f32..=500_f32);

        let max_hunger: f32 = rng.random_range(50.0..=200.0);
        let max_health: f32 = rng.random_range(50.0..=200.0);
        let max_energy: f32 = rng.random_range(100.0..=150.0);
        let hover_color: Color = Color::srgb(6.25, 9.4, 9.1);

        let is_male: bool = rng.random_bool(0.5);
        let start_color: Color = if is_male {
            MALE_COLOR
        } else {
            FEMALE_COLOR
        };

        //Bobble
        commands.spawn((
            Bobble {
                age: 10,
                gender: if is_male { 
                    BobbleGender::Male
                } else {
                    BobbleGender::Female
                },
            },
            Hunger {
                hunger: max_hunger,
                max_hunger: max_hunger,
            },
            Health {
                health: max_health,
                max_health: max_health,
                alive: true,
                timer: Timer::from_seconds(START_HEALING_TIME, TimerMode::Once),
            },
            Sprite {
                image: asset_server.load("human.png"),
                color: start_color,
                custom_size: Some(Vec2::new(PLAYER_SCALE, PLAYER_SCALE)),
                ..default()
            },
            Transform::from_xyz(x, y, 0.),
            Pickable {
                should_block_lower: true,
                is_hoverable: true, 
            },
        ))
        .observe(|trigger: On<Pointer<Click>>, query: Query<(&Hunger, &Health)>| {
            println!("Click");
            let clicked_entity = trigger.entity;

            if let Ok((hunger, health)) = query.get(clicked_entity) {
                println!("Hunger: {}, Health: {}", hunger.hunger, health.health);
            }
        })
        .observe(move |trigger: On<Pointer<Over>>, mut query: Query<&mut Sprite>| {
            if let Ok(mut sprite_handle) = query.get_mut(trigger.entity) {
                sprite_handle.color = hover_color;
            }
        })
        .observe(move |trigger: On<Pointer<Out>>, mut query: Query<&mut Sprite>| {
            if let Ok(mut sprite_handle) = query.get_mut(trigger.entity) {
                sprite_handle.color = start_color;
            }
        });
    }

    //Plant
    for _ in 1..=INITIAL_SPAWN {
        let x: f32 = rng.random_range(-500_f32..=500_f32);
        let y: f32 = rng.random_range(-500_f32..=500_f32);

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
            Collider,
            Transform::from_xyz(x, y, 0.),
        ));
    }
}

fn setup_ui(mut commands: Commands) {
    commands.spawn((
        Text::new("Hunger: 100\nHealth: 100\nEnergy: 100"),
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
    target_query: Single<(&Hunger, &Health, &Energy, &Movement), With<Target>>,
) {
    for mut text in text_query.iter_mut() {
        *text = Text::new(format!(
            "Hunger {:.2} Health {:.2} Energy {:.2} Velocity ({:.2}, {:.2})", 
            target_query.0.hunger, 
            target_query.1.health, 
            target_query.2.energy, 
            target_query.3.velocity.x, 
            target_query.3.velocity.y
        ));
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d, Bloom::NATURAL));
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

fn update_hunger(
    time: Res<Time>,
    mut hunger_query: Query<&mut Hunger>
) {
    hunger_query.iter_mut().for_each(|mut query| {
        query.hunger -= HUNGER_RATE * time.delta_secs();

        if query.hunger <= 0. {
            query.hunger = 0.;
        }
    });
}

fn update_energy(
    time: Res<Time>,
    mut targets: Query<(&mut Energy, &Movement), With<Bobble>>
) {
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

fn update_health(
    time: Res<Time>,
    mut targets: Query<(&mut Health, &Hunger), With<Hunger>>
) {
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

fn update_velocity(
    time: Res<Time>,
    mut targets: Query<(&Transform, &mut Movement)>
) {
    targets.iter_mut().for_each(|(transform, mut movement)| {
        let delta_x = transform.translation.x - movement.last_x;
        let delta_y = transform.translation.y - movement.last_y;
        movement.velocity.x = delta_x / time.delta_secs();
        movement.velocity.y = delta_y / time.delta_secs();

        movement.last_x = transform.translation.x;
        movement.last_y = transform.translation.y;
    });
}

fn despawn_dead(
    mut commands: Commands,
    query: Query<(Entity, &Health), With<Health>>,
) {
    query.iter().for_each(|(entity, entity_health)| {
        if !entity_health.alive {
            commands.entity(entity).despawn();
        }
    });
}

fn bobble_eating_collision(
    mut edible_collider_query: Query<(&mut Transform, &Edible), (With<Collider>, With<Edible>)>,
    mut bobble_collider_query: Query<(&Transform, &mut Hunger), (With<Collider>, With<Bobble>, With<Target>, Without<Edible>)>,
) {
    edible_collider_query.iter_mut().for_each(|(mut edible_transform, edible)| {
        bobble_collider_query.iter_mut().for_each(|(bobble_transform, mut hunger)| {
            let dist = edible_transform.translation.truncate().distance(bobble_transform.translation.truncate());
            if dist < COLLISION_DISTANCE {
                // "Despawn" eaten thing (Move it somewhere else)
                let mut rng = rand::rng();
                let x: f32 = rng.random_range(-500_f32..=500_f32);
                let y: f32 = rng.random_range(-500_f32..=500_f32);

                edible_transform.translation = Vec3::new(x, y, 0.);
                
                hunger.hunger += edible.nutrition_value;
                if hunger.hunger > hunger.max_hunger {
                    hunger.hunger = hunger.max_hunger;
                }
            }
        });
    });
}

fn bobble_reproducing_collision(
    mut bobble_query: Query<(&Transform, &Hunger, &Health, &Energy, &Bobble), (With<Reproducing>, With<Bobble>)>
) {
    let mut combinations = bobble_query.iter_combinations_mut();
    while let Some([
            (transform1, hunger1, health1, energy1, bobble1), 
            (transform2, hunger2, health2, energy2, bobble2)
        ]) = combinations.fetch_next() {
        if bobble1.gender != bobble2.gender { 
            // One is male, one is female
            let dist = transform1.translation.truncate().distance(transform2.translation.truncate());
            if dist < COLLISION_DISTANCE {
                // They're close enough...
                let mut rng = rand::rng();
                let reproductibility_score = 
            }
            // Now we combine them
        }
    }
}
