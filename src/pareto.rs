use crate::ast::Node;
use crate::genetic::{crossover, mutate_hoist, mutate_point, mutate_shrink, tournament_select};
use rand::Rng;

/// A solution in multi-objective space.
#[derive(Debug, Clone)]
pub struct MoSolution {
    pub genome: Node,
    pub objectives: Vec<f64>, // all minimized
    pub rank: usize,
    pub crowding_distance: f64,
}

/// Configuration for NSGA-II style multi-objective GP.
#[derive(Debug, Clone)]
pub struct NsgaConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub max_depth: usize,
    pub max_tree_size: usize,
    pub num_vars: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
}

impl Default for NsgaConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            max_generations: 50,
            max_depth: 6,
            max_tree_size: 100,
            num_vars: 1,
            tournament_size: 3,
            crossover_rate: 0.7,
            mutation_rate: 0.2,
        }
    }
}

/// Statistics for a generation in NSGA-II.
#[derive(Debug, Clone)]
pub struct NsgaStats {
    pub generation: usize,
    pub pareto_front_size: usize,
    pub best_obj0: f64,
    pub best_obj1: f64,
}

/// Returns true if `a` dominates `b` (all objectives <= and at least one <).
fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut at_least_one_better = false;
    for (ai, bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            return false;
        }
        if ai < bi {
            at_least_one_better = true;
        }
    }
    at_least_one_better
}

/// Compute non-dominated sorting ranks. Returns rank for each individual (0-indexed).
fn non_dominated_sort(objectives: &[Vec<f64>]) -> Vec<usize> {
    let n = objectives.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut ranks = vec![0usize; n];
    let mut front: Vec<usize> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&objectives[i], &objectives[j]) {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if dominates(&objectives[j], &objectives[i]) {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            ranks[i] = 0;
            front.push(i);
        }
    }

    let mut current_rank = 0;
    while !front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &front {
            ranks[i] = current_rank;
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        current_rank += 1;
        front = next_front;
    }

    ranks
}

/// Compute crowding distance for individuals within the same front.
fn crowding_distance(objectives: &[Vec<f64>], front_indices: &[usize]) -> Vec<f64> {
    let n = front_indices.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let num_objectives = objectives[front_indices[0]].len();
    let mut distances = vec![0.0f64; n];

    #[allow(clippy::needless_range_loop)]
    for m in 0..num_objectives {
        // Sort front by objective m
        let mut sorted: Vec<usize> = (0..n).collect();
        sorted.sort_by(|&a, &b| {
            objectives[front_indices[a]][m]
                .partial_cmp(&objectives[front_indices[b]][m])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary points get infinite distance
        distances[sorted[0]] = f64::INFINITY;
        distances[sorted[n - 1]] = f64::INFINITY;

        let obj_min = objectives[front_indices[sorted[0]]][m];
        let obj_max = objectives[front_indices[sorted[n - 1]]][m];
        let range = obj_max - obj_min;

        if range > 1e-10 {
            for i in 1..(n - 1) {
                let prev = objectives[front_indices[sorted[i - 1]]][m];
                let next = objectives[front_indices[sorted[i + 1]]][m];
                distances[sorted[i]] += (next - prev) / range;
            }
        }
    }

    distances
}

/// Run NSGA-II multi-objective GP evolution.
/// `objective_fns` is a slice of objective functions, each returning a value to minimize.
pub fn nsga2_evolve<R>(
    rng: &mut R,
    config: &NsgaConfig,
    objective_fns: &[&dyn Fn(&Node) -> f64],
) -> (Vec<MoSolution>, Vec<NsgaStats>)
where
    R: Rng,
{
    let num_obj = objective_fns.len();

    // Initialize population
    let mut genomes: Vec<Node> = (0..config.population_size)
        .map(|_| Node::random(rng, config.max_depth, config.num_vars))
        .collect();

    let mut stats = Vec::new();

    for gen in 0..config.max_generations {
        // Evaluate objectives
        let objectives: Vec<Vec<f64>> = genomes
            .iter()
            .map(|g| objective_fns.iter().map(|f| f(g)).collect())
            .collect();

        // Non-dominated sorting
        let ranks = non_dominated_sort(&objectives);

        // Group by rank
        let max_rank = *ranks.iter().max().unwrap_or(&0);
        let mut fronts: Vec<Vec<usize>> = vec![Vec::new(); max_rank + 1];
        for (i, &r) in ranks.iter().enumerate() {
            fronts[r].push(i);
        }

        // Compute crowding distances per front
        let mut crowd_dist = vec![0.0f64; genomes.len()];
        for front in &fronts {
            let dists = crowding_distance(&objectives, front);
            for (local_i, &global_i) in front.iter().enumerate() {
                crowd_dist[global_i] = dists[local_i];
            }
        }

        // Stats
        let pareto_front_size = fronts[0].len();
        let best_obj0 = objectives
            .iter()
            .map(|o| o[0])
            .fold(f64::INFINITY, f64::min);
        let best_obj1 = if num_obj > 1 {
            objectives
                .iter()
                .map(|o| o[1])
                .fold(f64::INFINITY, f64::min)
        } else {
            0.0
        };

        stats.push(NsgaStats {
            generation: gen,
            pareto_front_size,
            best_obj0,
            best_obj1,
        });

        // Build (Node, f64) for tournament selection using combined rank+crowding score
        let selection_fitness: Vec<(Node, f64)> = genomes
            .iter()
            .enumerate()
            .map(|(i, g)| {
                // Lower rank is better; for same rank, higher crowding is better
                // Encode as: rank * big_number - crowding_distance
                let score = ranks[i] as f64 * 1e6 - crowd_dist[i].min(1e5); // clamp infinity
                (g.clone(), score)
            })
            .collect();

        // Create offspring
        let mut offspring = Vec::with_capacity(config.population_size);

        while offspring.len() < config.population_size {
            let r: f64 = rng.gen();
            let child = if r < config.crossover_rate {
                let p1 = tournament_select(rng, &selection_fitness, config.tournament_size);
                let p2 = tournament_select(rng, &selection_fitness, config.tournament_size);
                let (c1, _) = crossover(rng, p1, p2);
                c1
            } else if r < config.crossover_rate + config.mutation_rate {
                let p = tournament_select(rng, &selection_fitness, config.tournament_size);
                match rng.gen_range(0..3) {
                    0 => mutate_point(rng, p, config.num_vars),
                    1 => mutate_hoist(rng, p),
                    _ => mutate_shrink(rng, p, config.num_vars),
                }
            } else {
                tournament_select(rng, &selection_fitness, config.tournament_size).clone()
            };

            if child.size() <= config.max_tree_size {
                offspring.push(child);
            } else {
                offspring.push(
                    tournament_select(rng, &selection_fitness, config.tournament_size).clone(),
                );
            }
        }

        genomes = offspring;
    }

    // Final evaluation and return Pareto front
    let objectives: Vec<Vec<f64>> = genomes
        .iter()
        .map(|g| objective_fns.iter().map(|f| f(g)).collect())
        .collect();
    let ranks = non_dominated_sort(&objectives);

    let mut front_indices: Vec<usize> = Vec::new();
    for (i, &r) in ranks.iter().enumerate() {
        if r == 0 {
            front_indices.push(i);
        }
    }
    let dists = crowding_distance(&objectives, &front_indices);

    let pareto_front: Vec<MoSolution> = front_indices
        .iter()
        .enumerate()
        .map(|(local_i, &global_i)| MoSolution {
            genome: genomes[global_i].clone(),
            objectives: objectives[global_i].clone(),
            rank: 0,
            crowding_distance: dists[local_i],
        })
        .collect();

    (pareto_front, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::Interpreter;
    use rand::SeedableRng;

    #[test]
    fn test_dominates() {
        assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0])); // neither dominates
        assert!(!dominates(&[2.0, 2.0], &[1.0, 1.0]));
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0])); // equal, no domination
    }

    #[test]
    fn test_non_dominated_sort() {
        let objectives = vec![
            vec![1.0, 5.0], // rank 0 (Pareto front)
            vec![5.0, 1.0], // rank 0
            vec![3.0, 3.0], // rank 0
            vec![4.0, 4.0], // rank 1 (dominated by [3,3])
            vec![6.0, 6.0], // rank 2
        ];
        let ranks = non_dominated_sort(&objectives);
        assert_eq!(ranks[0], 0);
        assert_eq!(ranks[1], 0);
        assert_eq!(ranks[2], 0);
        assert_eq!(ranks[3], 1);
        assert_eq!(ranks[4], 2);
    }

    #[test]
    fn test_crowding_distance() {
        let objectives = vec![vec![1.0, 5.0], vec![3.0, 3.0], vec![5.0, 1.0]];
        let front = vec![0, 1, 2];
        let dists = crowding_distance(&objectives, &front);
        // Boundary points should have infinite distance
        assert!(dists[0].is_infinite() || dists[2].is_infinite());
        // Middle point should have finite distance
        assert!(dists[1].is_finite());
    }

    #[test]
    fn test_nsga2_evolve() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = NsgaConfig {
            population_size: 50,
            max_generations: 10,
            max_depth: 4,
            num_vars: 1,
            ..NsgaConfig::default()
        };

        // Two objectives: accuracy on x^2, and program simplicity
        let obj_accuracy = |tree: &Node| -> f64 {
            let mut interp = Interpreter::default();
            let mut error = 0.0;
            for i in -3..=3 {
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
            error / 7.0
        };

        let obj_simplicity = |tree: &Node| -> f64 { tree.size() as f64 };

        let (pareto_front, stats) =
            nsga2_evolve(&mut rng, &config, &[&obj_accuracy, &obj_simplicity]);

        assert!(!pareto_front.is_empty());
        assert!(!stats.is_empty());
        // All Pareto front members should have rank 0
        for sol in &pareto_front {
            assert_eq!(sol.rank, 0);
            assert_eq!(sol.objectives.len(), 2);
        }
    }
}
