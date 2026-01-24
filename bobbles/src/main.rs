use avian2d::prelude::*;
use bevy::prelude::*;
use rand::Rng;
use std::f32::consts::FRAC_PI_3;
//use std::collections::{HashMap, HashSet};

// Neural Network Stuffs
//#[derive(Debug, Clone, Copy, PartialEq)]
//enum NodeType {
//    Input,
//    Hidden,
//    Output,
//}
//
//#[derive(Debug, Clone)]
//struct Connection {
//    from_idx: usize,
//    to_idx: usize,
//    weight: f32,
//    enabled: bool,
//    innovation: usize,
//}
//
//impl Connection {
//    fn new(from_idx: usize, to_idx: usize, weight: f32, innovation: usize) -> Self {
//        Self {
//            from_idx,
//            to_idx,
//            weight,
//            enabled: true,
//            innovation,
//        }
//    }
//}
//
//#[derive(Debug, Clone, Default)]
//struct Genome {
//    nodes: HashMap<usize, NodeType>,
//    connections: Vec<Connection>,
//    pub fitness: f32,
//}
//
//#[derive(Resource, Default)]
//struct InnovationTracker {
//    current_number: usize,
//    history: HashMap<(usize, usize), usize>, // (from, to) -> innovation_id
//}
//
//#[derive(Component)]
//struct NeuralNetwork {
//    nodes: Vec<NodeState>,
//    execution_order: Vec<usize>,
//    inputs_count: usize,
//    output_indices: Vec<usize>,
//}
//
//pub struct NodeState {
//    pub id: usize,
//    pub value: f32,
//    pub incoming: Vec<(usize, f32)>, // (index_in_nodes_vec, weight)
//    pub node_type: NodeType,
//}
//
//#[derive(Component)]
//struct Fitness(f64);
//
//#[derive(Resource, Default)]
//pub struct InnovationHistory {
//    pub map: HashMap<(usize, usize), usize>,
//    pub next_innovation: usize,
//    pub next_node_id: usize,
//}
//
//impl InnovationHistory {
//    pub fn get_innovation(&mut self, from: usize, to: usize) -> usize {
//        if let Some(&id) = self.map.get(&(from, to)) {
//            id
//        } else {
//            let id = self.next_innovation;
//            self.map.insert((from, to), id);
//            self.next_innovation += 1;
//            id
//        }
//    }
//}
//
//impl Genome {
//    pub fn compile(&self) -> NeuralNetwork {
//        let mut nodes_vec = Vec::new();
//        let mut id_to_idx = HashMap::new();
//
//        for (id, node_type) in &self.nodes {
//            id_to_idx.insert(*id, nodes_vec.len());
//            nodes_vec.push(NodeState {
//                id: *id,
//                value: 0.0,
//                incoming: Vec::new(),
//                node_type: *node_type,
//            });
//        }
//
//        for conn in self.connections.iter().filter(|c| c.enabled) {
//            let to_idx = id_to_idx[&conn.to_idx];
//            let from_idx = id_to_idx[&conn.from_idx];
//            nodes_vec[to_idx].incoming.push((from_idx, conn.weight));
//        }
//
//        let mut execution_order = Vec::new();
//        let mut visited = HashSet::new();
//
//        fn visit(
//            idx: usize,
//            nodes: &Vec<NodeState>,
//            visited: &mut HashSet<usize>,
//            order: &mut Vec<usize>,
//            id_to_idx: &HashMap<usize, usize>,
//        ) {
//            if visited.contains(&idx) || nodes[idx].node_type == NodeType::Input {
//                return;
//            }
//            for (from_idx, _) in &nodes[idx].incoming {
//                visit(*from_idx, nodes, visited, order, id_to_idx);
//            }
//            visited.insert(idx);
//            order.push(idx);
//        }
//
//        let output_indices: Vec<usize> = nodes_vec
//            .iter()
//            .enumerate()
//            .filter(|(_, n)| n.node_type == NodeType::Output)
//            .map(|(i, _)| i)
//            .collect();
//
//        for &out_idx in &output_indices {
//            visit(
//                out_idx,
//                &nodes_vec,
//                &mut visited,
//                &mut execution_order,
//                &id_to_idx,
//            );
//        }
//
//        NeuralNetwork {
//            nodes: nodes_vec,
//            execution_order,
//            inputs_count: self
//                .nodes
//                .values()
//                .filter(|&&t| t == NodeType::Input)
//                .count(),
//            output_indices,
//        }
//    }
//}
//
//impl NeuralNetwork {
//    pub fn activate(&mut self, inputs: &[f32]) -> Vec<f32> {
//        let mut input_ptr = 0;
//        for node in &mut self.nodes {
//            if node.node_type == NodeType::Input {
//                node.value = inputs[input_ptr];
//                input_ptr += 1;
//            }
//        }
//
//        for &idx in &self.execution_order {
//            let sum: f32 = self.nodes[idx]
//                .incoming
//                .iter()
//                .map(|(from_idx, weight)| self.nodes[*from_idx].value * weight)
//                .sum();
//            self.nodes[idx].value = sum.tanh(); // Using Tanh for -1 to 1 output
//        }
//
//        self.output_indices
//            .iter()
//            .map(|&i| self.nodes[i].value)
//            .collect()
//    }
//}

// --- 5. MUTATION LOGIC ---

//impl Genome {
//    pub fn mutate(&mut self, history: &mut InnovationHistory) {
//        let mut rng = rand::rng();
//        let mutation_type: f32 = rng.random();
//
//        if mutation_type < 0.8 {
//            // 80% Weight Mutation
//            for conn in &mut self.connections {
//                if rng.random_bool(0.9) {
//                    conn.weight += rng.random_range(-0.1..0.1); // Nudge
//                } else {
//                    conn.weight = rng.random_range(-1.0..1.0); // Reset
//                }
//            }
//        } else if mutation_type < 0.85 {
//            // 5% Add Connection
//            let keys: Vec<usize> = self.nodes.keys().cloned().collect();
//            let from_idx = *keys.choose(&mut rng).unwrap();
//            let to_idx = *keys.choose(&mut rng).unwrap();
//
//            // Basic check: don't connect to an input, and don't connect to self
//            if self.nodes[&to_idx] != NodeType::Input && from_idx != to_idx {
//                let innov = history.get_innovation(from_idx, to_idx);
//                self.connections.push(Connection {
//                    from_idx,
//                    to_idx,
//                    weight: rng.random_range(-1.0..1.0),
//                    enabled: true,
//                    innovation: innov,
//                });
//            }
//        } else if mutation_type < 0.88 {
//            // 3% Add Node
//            if let Some(conn) = self
//                .connections
//                .iter_mut()
//                .filter(|c| c.enabled)
//                .choose(&mut rng)
//            {
//                conn.enabled = false;
//                let new_id = history.next_node_id;
//                history.next_node_id += 1;
//
//                self.nodes.insert(new_id, NodeType::Hidden);
//
//                // Add two connections to replace the old one
//                let innov1 = history.get_innovation(conn.from_idx, new_id);
//                let innov2 = history.get_innovation(new_id, conn.to_idx);
//
//                //self.connections.push(Connection { from_idx: conn.from_idx, to_idx: new_id, weight: 1.0, enabled: true, innovation: innov1 });
//                //self.connections.push(Connection { from_idx: new_id, to_idx: conn.to_idx, weight: conn.weight, enabled: true, innovation: innov2 });
//            }
//        }
//    }
//}

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
const COLLISION_DISTANCE: f32 = 20.;
const START_HEALING_TIME: f32 = 10.;
const START_RESTING_TIME: f32 = 2.;
const LIFE_EXPECTANCY: f32 = 70.;

const PREGNANCY_TIME: f32 = 20.;
const REPRODUCTION_AGE: f32 = 18.0;

const VISION_DISTANCE: f32 = 200.0;
const VISION_FOV: f32 = 2.0 * FRAC_PI_3; // 120 degree fov
const MOVEMENT_PER_TICK: f32 = 100.0;

const MALE_COLOR: Color = Color::srgb(0., 0., 1.);
const FEMALE_COLOR: Color = Color::srgb(1., 0., 1.);
const HOVER_COLOR: Color = Color::srgb(1., 0., 0.);

#[derive(PartialEq)]
enum BobbleGender {
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
struct Bobble {
    age: f32,
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
    is_male: bool,
}

#[derive(Component)]
struct Vision {
    heading: Vec2,
    seeing: Vec<String>,
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, PhysicsPlugins::default()))
        .add_plugins(MeshPickingPlugin)
        .add_systems(Startup, (setup_scene, setup_camera, setup_ui))
        .add_systems(
            Update,
            (
                ((update_health, update_hunger, update_energy), despawn_dead).chain(),
                (move_target, update_camera).chain(),
                bobble_eating_collision,
                bobble_reproducing_collision,
                update_ui,
                update_heading,
                update_reproduction,
                update_age,
                update_vision,
                move_bobble,
            ),
        )
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
    commands
        .spawn((
            Target,
            Transform::from_xyz(0., 0., 0.),
            Bobble {
                age: 10.,
                gender: BobbleGender::Male,
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
            },
            Name {
                name: "Player".to_string(),
            },
            VisionTarget,
        ))
        .insert(Sensor);

    let mut rng = rand::rng();
    for _ in 1..=INITIAL_SPAWN {
        let x: f32 = rng.random_range(-500_f32..=500_f32);
        let y: f32 = rng.random_range(-500_f32..=500_f32);

        let max_hunger: f32 = rng.random_range(50.0..=200.0);
        let max_health: f32 = rng.random_range(50.0..=200.0);
        let max_energy: f32 = rng.random_range(100.0..=150.0);

        let is_male: bool = rng.random_bool(0.5);

        //Bobble
        spawn_bobble(
            &mut commands,
            &asset_server,
            Vec2::new(x, y),
            is_male,
            max_hunger,
            max_health,
            max_energy,
            10.0,
        );
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
    target_query: Single<(&Hunger, &Health, &Energy, &Movement, &Vision), With<Target>>,
) {
    for mut text in text_query.iter_mut() {
        *text = Text::new(format!(
            "Hunger {:.2} Health {:.2} Energy {:.2} Velocity ({:.2}, {:.2}) Seeing: [{:?}]",
            target_query.0.hunger,
            target_query.1.health,
            target_query.2.energy,
            target_query.3.velocity.x,
            target_query.3.velocity.y,
            target_query.4.seeing,
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

fn update_energy(time: Res<Time>, mut targets: Query<(&mut Energy, &Movement), With<Bobble>>) {
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

fn update_age(time: Res<Time>, mut targets: Query<(&mut Bobble, &mut Health)>) {
    targets.iter_mut().for_each(|(mut bobble, mut health)| {
        if health.alive {
            bobble.age += time.delta_secs();

            if bobble.age > LIFE_EXPECTANCY {
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
    mut targets: Query<(&Transform, &mut Reproducing)>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    targets.iter_mut().for_each(|(transform, mut reproducing)| {
        if reproducing.pregnant {
            reproducing.timer.tick(time.delta());
            if reproducing.timer.is_finished() {
                spawn_bobble(
                    &mut commands,
                    &asset_server,
                    Vec2::new(transform.translation.x, transform.translation.y),
                    reproducing.is_male,
                    reproducing.child_hunger,
                    reproducing.child_health,
                    reproducing.child_energy,
                    0.0,
                );

                reproducing.pregnant = false;
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

fn bobble_eating_collision(
    mut collision_event_reader: MessageReader<CollisionStart>,
    mut edible_collider_query: Query<(&mut Transform, &Edible)>,
    mut bobble_collider_query: Query<&mut Hunger, With<Bobble>>,
) {
    for CollisionStart {
        collider1: e1,
        collider2: e2,
        ..
    } in collision_event_reader.read()
    {
        let collision_pair = if let Ok(hunger) = bobble_collider_query.get_mut(*e1) {
            edible_collider_query
                .get_mut(*e2)
                .map(|edible| (hunger, edible))
                .ok()
        } else if let Ok(hunger) = bobble_collider_query.get_mut(*e2) {
            edible_collider_query
                .get_mut(*e1)
                .map(|edible| (hunger, edible))
                .ok()
        } else {
            None
        };

        if let Some((mut hunger, (mut edible_transform, edible))) = collision_pair {
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
    }
}

fn bobble_reproducing_collision(
    mut collision_event_reader: MessageReader<CollisionStart>,
    mut bobble_query: Query<(
        &Transform,
        &Hunger,
        &Health,
        &Energy,
        &Bobble,
        &mut Reproducing,
    )>,
) {
    for CollisionStart {
        collider1: e1,
        collider2: e2,
        ..
    } in collision_event_reader.read()
    {
        if let Ok([b1, b2]) = bobble_query.get_many_mut([*e1, *e2]) {
            let (transform1, hunger1, health1, energy1, bobble1, reproducing1) = b1;
            let (transform2, hunger2, health2, energy2, bobble2, reproducing2) = b2;

            if bobble1.gender != bobble2.gender
                && !reproducing1.pregnant
                && !reproducing2.pregnant
                && bobble1.age > REPRODUCTION_AGE
                && bobble2.age > REPRODUCTION_AGE
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
                        let mut reproducing = if bobble1.gender == BobbleGender::Female {
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
                        let is_male: bool = rng.random_bool(0.5);
                        reproducing.pregnant = true;
                        reproducing.child_health = child_health;
                        reproducing.child_hunger = child_hunger;
                        reproducing.child_energy = child_energy;
                        reproducing.is_male = is_male;
                        println!("Pregante");
                    }
                }
            }
        }
    }
}

fn spawn_bobble(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    spawn_loc: Vec2,
    is_male: bool,
    hunger: f32,
    health: f32,
    energy: f32,
    age: f32,
) {
    let start_color: Color = if is_male { MALE_COLOR } else { FEMALE_COLOR };

    commands
        .spawn((
            Bobble {
                age: age,
                gender: if is_male {
                    BobbleGender::Male
                } else {
                    BobbleGender::Female
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
                is_male: false,
            },
            Collider::circle(5.0),
            CollisionEventsEnabled,
            CollidingEntities::default(),
            Vision {
                heading: Vec2::new(1.0, 0.0),
                seeing: Vec::new(),
            },
            VisionTarget,
            Name {
                name: "Bobble".to_string(),
            },
            Movement {
                velocity: Vec2::new(0.0, 0.0),
                last_x: 0.0,
                last_y: 0.0,
            },
        ))
        .observe(
            |trigger: On<Pointer<Click>>, query: Query<(&Hunger, &Health)>| {
                let clicked_entity = trigger.entity;

                if let Ok((hunger, health)) = query.get(clicked_entity) {
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

fn move_bobble(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &Movement), (With<Bobble>, Without<Target>)>,
) {
    let mut rng = rand::rng();

    query.iter_mut().for_each(|(mut transform, _movement)| {
        let dir = rng.random_range(1..=8); // N, NE, E, SE, S, SW, W, NW

        let movement_unit = MOVEMENT_PER_TICK * time.delta_secs();

        match dir {
            1 => {
                transform.translation += Vec3::new(0.0, movement_unit, 0.0);
            }
            2 => {
                transform.translation += Vec3::new(movement_unit, movement_unit, 0.0);
            }
            3 => {
                transform.translation += Vec3::new(movement_unit, 0.0, 0.0);
            }
            4 => {
                transform.translation += Vec3::new(movement_unit, -movement_unit, 0.0);
            }
            5 => {
                transform.translation += Vec3::new(0.0, -movement_unit, 0.0);
            }
            6 => {
                transform.translation += Vec3::new(-movement_unit, -movement_unit, 0.0);
            }
            7 => {
                transform.translation += Vec3::new(-movement_unit, 0.0, 0.0);
            }
            8 => {
                transform.translation += Vec3::new(-movement_unit, movement_unit, 0.0);
            }
            _ => {
                println!("How did you get here");
            }
        }
    });
}

// TODO Add raycast for field of vision
fn update_vision(
    spatial_query: SpatialQuery,
    mut viewer_query: Query<(Entity, &Transform, &mut Vision)>,
    vision_target_query: Query<(Entity, &Transform, &Name), With<VisionTarget>>,
    //mut gizmos: Gizmos,
) {
    viewer_query
        .iter_mut()
        .for_each(|(viewer_ent, viewer_transform, mut viewer_vision)| {
            let viewer_pos = viewer_transform.translation.truncate();
            viewer_vision.seeing.clear();

            vision_target_query
                .iter()
                .for_each(|(target_ent, target_transform, target_name)| {
                    let target_pos = target_transform.translation.truncate();
                    let to_target = target_pos - viewer_pos;
                    let distance = to_target.length();

                    if distance < VISION_DISTANCE
                        && viewer_vision.heading.angle_to(to_target).abs() < VISION_FOV
                    {
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
                                viewer_vision.seeing.push(target_name.name.clone());
                            }
                        }
                    }
                });
        });
}
