use crate::ast::Node;
use rand::Rng;

/// Subtree crossover: swap a random subtree between two parents.
pub fn crossover<R: Rng>(rng: &mut R, parent1: &Node, parent2: &Node) -> (Node, Node) {
    let size1 = parent1.size();
    let size2 = parent2.size();

    let point1 = rng.gen_range(0..size1);
    let point2 = rng.gen_range(0..size2);

    let subtree1 = parent1.get_node(point1).unwrap().clone();
    let subtree2 = parent2.get_node(point2).unwrap().clone();

    let mut child1 = parent1.clone();
    let mut child2 = parent2.clone();
    child1.replace_node(point1, subtree2);
    child2.replace_node(point2, subtree1);

    (child1, child2)
}

/// Point mutation: replace a random node with a new random subtree.
pub fn mutate_point<R: Rng>(rng: &mut R, tree: &Node, num_vars: usize) -> Node {
    let size = tree.size();
    let point = rng.gen_range(0..size);
    let replacement = Node::random(rng, 2, num_vars);
    let mut result = tree.clone();
    result.replace_node(point, replacement);
    result
}

/// Hoist mutation: replace tree with one of its subtrees (reduces bloat).
pub fn mutate_hoist<R: Rng>(rng: &mut R, tree: &Node) -> Node {
    let size = tree.size();
    if size <= 1 {
        return tree.clone();
    }
    let point = rng.gen_range(1..size); // skip root
    tree.get_node(point).unwrap().clone()
}

/// Shrink mutation: replace a random subtree with a terminal.
pub fn mutate_shrink<R: Rng>(rng: &mut R, tree: &Node, num_vars: usize) -> Node {
    let size = tree.size();
    let point = rng.gen_range(0..size);
    let terminal = Node::random(rng, 0, num_vars);
    let mut result = tree.clone();
    result.replace_node(point, terminal);
    result
}

/// Tournament selection: pick the best individual from a random subset.
pub fn tournament_select<'a, R: Rng>(
    rng: &mut R,
    population: &'a [(Node, f64)],
    tournament_size: usize,
) -> &'a Node {
    let mut best_idx = rng.gen_range(0..population.len());
    let mut best_fitness = population[best_idx].1;

    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..population.len());
        if population[idx].1 < best_fitness {
            // lower is better (minimization)
            best_idx = idx;
            best_fitness = population[idx].1;
        }
    }
    &population[best_idx].0
}

/// Individual with metadata.
#[derive(Debug, Clone)]
pub struct Individual {
    pub genome: Node,
    pub fitness: f64,
}

/// Configuration for the GP engine.
#[derive(Debug, Clone)]
pub struct GpConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub max_depth: usize,
    pub max_tree_size: usize,
    pub num_vars: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elitism: usize,
}

impl Default for GpConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            max_generations: 100,
            max_depth: 6,
            max_tree_size: 100,
            num_vars: 1,
            tournament_size: 5,
            crossover_rate: 0.7,
            mutation_rate: 0.2,
            elitism: 2,
        }
    }
}

/// Statistics for a generation.
#[derive(Debug, Clone)]
pub struct GenStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub avg_size: f64,
    pub best_program: String,
}

/// Run the GP evolution loop.
pub fn evolve<R, F>(rng: &mut R, config: &GpConfig, fitness_fn: &F) -> (Node, f64, Vec<GenStats>)
where
    R: Rng,
    F: Fn(&Node) -> f64,
{
    // Initialize population
    let mut population: Vec<(Node, f64)> = (0..config.population_size)
        .map(|_| {
            let tree = Node::random(rng, config.max_depth, config.num_vars);
            let fit = fitness_fn(&tree);
            (tree, fit)
        })
        .collect();

    let mut hall_of_fame: Option<(Node, f64)> = None;
    let mut stats = Vec::new();

    for gen in 0..config.max_generations {
        // Sort by fitness (lower is better)
        population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Stats
        let best_fit = population[0].1;
        let avg_fit = population.iter().map(|(_, f)| f).sum::<f64>() / population.len() as f64;
        let avg_size =
            population.iter().map(|(t, _)| t.size() as f64).sum::<f64>() / population.len() as f64;

        stats.push(GenStats {
            generation: gen,
            best_fitness: best_fit,
            avg_fitness: avg_fit,
            avg_size,
            best_program: population[0].0.to_expr(),
        });

        // Update hall of fame
        if hall_of_fame.is_none() || best_fit < hall_of_fame.as_ref().unwrap().1 {
            hall_of_fame = Some(population[0].clone());
        }

        // Early termination
        if best_fit < 1e-10 {
            break;
        }

        // Create next generation
        let mut next_gen = Vec::with_capacity(config.population_size);

        // Elitism
        for (tree, _) in population.iter().take(config.elitism.min(population.len())) {
            next_gen.push(tree.clone());
        }

        // Fill rest with crossover and mutation
        while next_gen.len() < config.population_size {
            let r: f64 = rng.gen();
            let child = if r < config.crossover_rate {
                let p1 = tournament_select(rng, &population, config.tournament_size);
                let p2 = tournament_select(rng, &population, config.tournament_size);
                let (c1, _) = crossover(rng, p1, p2);
                c1
            } else if r < config.crossover_rate + config.mutation_rate {
                let p = tournament_select(rng, &population, config.tournament_size);
                match rng.gen_range(0..3) {
                    0 => mutate_point(rng, p, config.num_vars),
                    1 => mutate_hoist(rng, p),
                    _ => mutate_shrink(rng, p, config.num_vars),
                }
            } else {
                // Reproduction
                tournament_select(rng, &population, config.tournament_size).clone()
            };

            // Bloat control
            if child.size() <= config.max_tree_size {
                next_gen.push(child);
            } else {
                // If too big, just copy a parent
                next_gen.push(tournament_select(rng, &population, config.tournament_size).clone());
            }
        }

        // Evaluate new population
        population = next_gen
            .into_iter()
            .map(|tree| {
                let fit = fitness_fn(&tree);
                (tree, fit)
            })
            .collect();
    }

    let (best, fit) = hall_of_fame.unwrap_or_else(|| population[0].clone());
    (best, fit, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_crossover() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let p1 = Node::BinOp(
            crate::ast::BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::FloatConst(1.0)),
        );
        let p2 = Node::BinOp(
            crate::ast::BinOp::Mul,
            Box::new(Node::FloatConst(2.0)),
            Box::new(Node::Var(0)),
        );
        let (c1, c2) = crossover(&mut rng, &p1, &p2);
        assert!(c1.size() > 0);
        assert!(c2.size() > 0);
    }

    #[test]
    fn test_mutations() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree = Node::BinOp(
            crate::ast::BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::FloatConst(1.0)),
        );

        let m1 = mutate_point(&mut rng, &tree, 1);
        assert!(m1.size() > 0);

        let m2 = mutate_hoist(&mut rng, &tree);
        assert!(m2.size() > 0);
        assert!(m2.size() <= tree.size());

        let m3 = mutate_shrink(&mut rng, &tree, 1);
        assert!(m3.size() > 0);
    }

    #[test]
    fn test_tournament_selection() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let pop = vec![
            (Node::IntConst(1), 10.0),
            (Node::IntConst(2), 5.0),
            (Node::IntConst(3), 1.0),
        ];
        // With enough samples, should tend to pick the best
        let mut best_count = 0;
        for _ in 0..100 {
            let selected = tournament_select(&mut rng, &pop, 3);
            if *selected == Node::IntConst(3) {
                best_count += 1;
            }
        }
        assert!(best_count > 50); // should pick best most of the time
    }

    #[test]
    fn test_evolve_simple() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GpConfig {
            population_size: 50,
            max_generations: 20,
            max_depth: 4,
            num_vars: 1,
            ..GpConfig::default()
        };

        // Try to evolve x^2: fitness = sum of squared errors on sample points
        let fitness = |tree: &Node| -> f64 {
            let mut interp = crate::interpreter::Interpreter::default();
            let mut error = 0.0;
            for i in -5..=5 {
                let x = i as f64;
                interp.reset();
                match interp.eval(tree, &[x]) {
                    Ok(val) => {
                        let diff = val.to_f64() - x * x;
                        error += diff * diff;
                    }
                    Err(_) => error += 1e6,
                }
            }
            error
        };

        let (best, best_fit, stats) = evolve(&mut rng, &config, &fitness);
        assert!(!stats.is_empty());
        // Best should be better than random
        assert!(best_fit < 1e6);
        assert!(best.size() > 0);
    }
}
