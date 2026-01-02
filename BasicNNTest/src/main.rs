use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
enum NodeType {
    Input,
    Hidden,
    Output,
}

#[derive(Debug, Clone)]
struct Node {
    id: usize,
    value: f64,
    node_type: NodeType,
}

impl Node {
    fn new(id: usize, node_type: NodeType) -> Self {
        Self {
            id,
            value: 0.0,
            node_type,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // Following the JS logic: inputs return value, others sum inputs.
    // Note: The JS version stores 'sum' in 'value' but returns 'sigmoid'.
    // In the JS 'forward' loop, the return value of compute is actually ignored,
    // so 'value' remains the raw sum. I will replicate that behavior.
    fn compute(&mut self, incoming_sum: f64) -> f64 {
        if self.node_type == NodeType::Input {
            return self.value;
        }
        self.value = incoming_sum;
        Self::sigmoid(self.value)
    }
}

#[derive(Debug, Clone)]
struct Connection {
    from_idx: usize,
    to_idx: usize,
    weight: f64,
    enabled: bool,
}

impl Connection {
    fn new(from_idx: usize, to_idx: usize, weight: f64) -> Self {
        Self {
            from_idx,
            to_idx,
            weight,
            enabled: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Genome {
    nodes: Vec<Node>,
    connections: Vec<Connection>,
}

impl Genome {
    fn add_node(&mut self, node_type: NodeType) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::new(id, node_type));
        id
    }

    fn add_connection(&mut self, from: usize, to: usize, weight: f64) {
        self.connections.push(Connection::new(from, to, weight));
    }

    fn forward(&mut self, inputs: &[f64]) -> Vec<f64> {
        // 1. Assign inputs
        let mut input_ptr = 0;
        for node in self.nodes.iter_mut() {
            if node.node_type == NodeType::Input && input_ptr < inputs.len() {
                node.value = inputs[input_ptr];
                input_ptr += 1;
            }
        }

        // 2. Compute nodes 
        // We use indices to satisfy the borrow checker (reading values while mutating others)
        for i in 0..self.nodes.len() {
            if self.nodes[i].node_type == NodeType::Input {
                continue;
            }

            let sum: f64 = self.connections.iter()
                .filter(|c| c.enabled && c.to_idx == i)
                .map(|c| c.weight * self.nodes[c.from_idx].value)
                .sum();

            self.nodes[i].compute(sum);
        }

        // 3. Collect outputs
        self.nodes.iter()
            .filter(|n| n.node_type == NodeType::Output)
            .map(|n| n.value)
            .collect()
    }
}

struct NeatConfig {
    population_size: usize,
    mutate_weight_chance: f64,
    new_connection_chance: f64,
    new_node_chance: f64,
}

struct Neat {
    population: Vec<Genome>,
    config: NeatConfig,
    generation: usize,
}

impl Neat {
    fn new(config: NeatConfig) -> Self {
        let mut population = Vec::new();
        for _ in 0..config.population_size {
            population.push(Self::create_initial_genome());
        }
        Self {
            population,
            config,
            generation: 0,
        }
    }

    fn create_initial_genome() -> Genome {
        let mut genome = Genome::default();
        let mut rng = rand::rng();
        let i1 = genome.add_node(NodeType::Input);
        let i2 = genome.add_node(NodeType::Input);
        let o = genome.add_node(NodeType::Output);
        
        genome.add_connection(i1, o, rng.random_range(-1.0..1.0));
        genome.add_connection(i2, o, rng.random_range(-1.0..1.0));
        genome
    }

    fn mutate(&self, genome: &mut Genome) {
        let mut rng = rand::rng();

        // Mutate Weights
        if rng.random_bool(self.config.mutate_weight_chance) && !genome.connections.is_empty() {
            let idx = rng.random_range(0..genome.connections.len());
            genome.connections[idx].weight += rng.random_range(-0.5..0.5);
        }

        // New Connection
        if rng.random_bool(self.config.new_connection_chance) {
            let n1_idx = rng.random_range(0..genome.nodes.len());
            let n2_idx = rng.random_range(0..genome.nodes.len());
            
            if genome.nodes[n1_idx].node_type != genome.nodes[n2_idx].node_type {
                genome.add_connection(n1_idx, n2_idx, rng.random_range(-1.0..1.0));
            }
        }

        // New Node
        if rng.random_bool(self.config.new_node_chance) && !genome.connections.is_empty() {
            let conn_idx = rng.random_range(0..genome.connections.len());
            genome.connections[conn_idx].enabled = false;

            let from = genome.connections[conn_idx].from_idx;
            let to = genome.connections[conn_idx].to_idx;
            let old_weight = genome.connections[conn_idx].weight;

            let middle_idx = genome.add_node(NodeType::Hidden);
            genome.add_connection(from, middle_idx, 1.0);
            genome.add_connection(middle_idx, to, old_weight);
        }
    }

    fn compute_fitness(genome: &mut Genome) -> f64 {
        let cases = [
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ];

        let mut total_error = 0.0;
        for (inputs, expected) in cases {
            let output = genome.forward(&inputs);
            let pred = output.get(0).unwrap_or(&0.0);
            total_error += (pred - expected).powi(2);
        }

        1.0 / (1.0 + total_error)
    }

    fn evolve(&mut self) {
        let fitnesses: Vec<f64> = self.population.iter_mut()
            .map(|g| Neat::compute_fitness(g))
            .collect();
        
        let total_fitness: f64 = fitnesses.iter().sum();
        println!("Generation {}: Total fitness: {}", self.generation, total_fitness);

        let mut new_population = Vec::new();
        let mut rng = rand::rng();

        while new_population.len() < self.config.population_size {
            // Roulette selection
            let pick = rng.random_range(0.0..total_fitness.max(0.1));
            let mut current = 0.0;
            for (i, &f) in fitnesses.iter().enumerate() {
                current += f;
                if current > pick {
                    let offspring = self.reproduce(&self.population[i]);
                    new_population.push(offspring);
                    break;
                }
            }
            // Safety break for zero fitness
            if total_fitness <= 0.0 { new_population.push(Self::create_initial_genome()); }
        }

        self.population = new_population;
        self.generation += 1;
    }

    fn reproduce(&self, parent: &Genome) -> Genome {
        let mut offspring = parent.clone();
        self.mutate(&mut offspring);
        offspring
    }
}

fn main() {
    let config = NeatConfig {
        population_size: 100,
        mutate_weight_chance: 0.8,
        new_connection_chance: 0.05,
        new_node_chance: 0.03,
    };

    let mut neat = Neat::new(config);
    let max_generations = 100;
    let target_fitness = 0.99;

    let mut best_genome: Option<Genome> = None;
    let mut best_fitness = 0.0;

    for _ in 0..max_generations {
        neat.evolve();

        for genome in neat.population.iter_mut() {
            let f = Neat::compute_fitness(genome);
            if f > best_fitness {
                best_fitness = f;
                best_genome = Some(genome.clone());
            }
        }

        if best_fitness >= target_fitness {
            break;
        }
    }

    if let Some(mut best) = best_genome {
        println!("Best Fitness: {}", best_fitness);
        let test_cases = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
        for tc in test_cases {
            let res = best.forward(&tc);
            println!("In: {:?}, Out: {:?}", tc, res);
        }
    }
}
