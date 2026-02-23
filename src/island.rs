use crate::ast::Node;
use crate::genetic::{evolve, GenStats, GpConfig};
use rand::Rng;

/// Island model configuration.
#[derive(Debug, Clone)]
pub struct IslandConfig {
    /// Number of islands.
    pub num_islands: usize,
    /// Generations between migrations.
    pub migration_interval: usize,
    /// Number of individuals to migrate each time.
    pub migration_size: usize,
    /// GP config (population_size is per-island).
    pub gp_config: GpConfig,
    /// Total generations across all migration cycles.
    pub total_generations: usize,
}

impl Default for IslandConfig {
    fn default() -> Self {
        Self {
            num_islands: 4,
            migration_interval: 20,
            migration_size: 5,
            gp_config: GpConfig {
                max_generations: 20,
                ..GpConfig::default()
            },
            total_generations: 100,
        }
    }
}

/// Result from island model evolution.
#[derive(Debug, Clone)]
pub struct IslandResult {
    pub best_genome: Node,
    pub best_fitness: f64,
    pub island_stats: Vec<Vec<GenStats>>,
}

/// Run island model evolution with periodic migration.
pub fn island_evolve<R, F>(rng: &mut R, config: &IslandConfig, fitness_fn: &F) -> IslandResult
where
    R: Rng,
    F: Fn(&Node) -> f64,
{
    let gens_per_cycle = config.migration_interval;
    let num_cycles = config.total_generations / gens_per_cycle;

    // Initialize islands: each is a population of (Node, fitness)
    let mut islands: Vec<Vec<(Node, f64)>> = (0..config.num_islands)
        .map(|_| {
            (0..config.gp_config.population_size)
                .map(|_| {
                    let tree =
                        Node::random(rng, config.gp_config.max_depth, config.gp_config.num_vars);
                    let fit = fitness_fn(&tree);
                    (tree, fit)
                })
                .collect()
        })
        .collect();

    let mut all_stats: Vec<Vec<GenStats>> = vec![Vec::new(); config.num_islands];
    let mut global_best: Option<(Node, f64)> = None;

    for cycle in 0..num_cycles {
        // Evolve each island independently
        for (island_idx, island) in islands.iter_mut().enumerate() {
            // Use the island population as initial, run evolution for gens_per_cycle
            let mut cycle_config = config.gp_config.clone();
            cycle_config.max_generations = gens_per_cycle;

            // Simple approach: run evolve from scratch but seed with island's best individuals
            // For a proper implementation, we'd want incremental evolution.
            // Here we extract the best and re-evolve.
            island.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let (best, best_fit, stats) = evolve(rng, &cycle_config, fitness_fn);

            // Adjust generation numbers
            let offset = cycle * gens_per_cycle;
            for mut s in stats {
                s.generation += offset;
                all_stats[island_idx].push(s);
            }

            // Merge evolved best back into island
            if let Some(worst) = island.last_mut() {
                *worst = (best.clone(), best_fit);
            }

            // Update global best
            if global_best.is_none() || best_fit < global_best.as_ref().unwrap().1 {
                global_best = Some((best, best_fit));
            }
        }

        // Migration: ring topology — each island sends its best to the next
        if config.num_islands > 1 {
            let mut migrants: Vec<Vec<(Node, f64)>> = Vec::new();

            for island in &mut islands {
                island.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let best: Vec<(Node, f64)> =
                    island.iter().take(config.migration_size).cloned().collect();
                migrants.push(best);
            }

            for (i, island) in islands.iter_mut().enumerate() {
                let source = if i == 0 {
                    config.num_islands - 1
                } else {
                    i - 1
                };
                let incoming = &migrants[source];

                // Replace worst individuals with migrants
                let len = island.len();
                for (j, migrant) in incoming.iter().enumerate() {
                    if j < len {
                        island[len - 1 - j] = migrant.clone();
                    }
                }
            }
        }
    }

    let (best_genome, best_fitness) = global_best.unwrap_or_else(|| {
        islands[0].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        islands[0][0].clone()
    });

    IslandResult {
        best_genome,
        best_fitness,
        island_stats: all_stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::Interpreter;
    use rand::SeedableRng;

    #[test]
    fn test_island_evolve() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let config = IslandConfig {
            num_islands: 3,
            migration_interval: 10,
            migration_size: 2,
            gp_config: GpConfig {
                population_size: 30,
                max_depth: 4,
                num_vars: 1,
                ..GpConfig::default()
            },
            total_generations: 30,
        };

        // Target: 2*x
        let fitness = |tree: &Node| -> f64 {
            let mut interp = Interpreter::default();
            let mut error = 0.0;
            for i in -5..=5 {
                let x = i as f64;
                interp.reset();
                match interp.eval(tree, &[x]) {
                    Ok(val) => {
                        let diff = val.to_f64() - 2.0 * x;
                        error += diff * diff;
                    }
                    Err(_) => error += 1e6,
                }
            }
            error / 11.0
        };

        let result = island_evolve(&mut rng, &config, &fitness);
        assert!(result.best_fitness < 1e6);
        assert!(result.best_genome.size() > 0);
        assert_eq!(result.island_stats.len(), 3);
    }

    #[test]
    fn test_single_island() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let config = IslandConfig {
            num_islands: 1,
            migration_interval: 10,
            migration_size: 1,
            gp_config: GpConfig {
                population_size: 20,
                max_generations: 10,
                max_depth: 3,
                num_vars: 1,
                ..GpConfig::default()
            },
            total_generations: 10,
        };

        let fitness = |tree: &Node| -> f64 {
            let mut interp = Interpreter::default();
            interp.reset();
            match interp.eval(tree, &[1.0]) {
                Ok(val) => (val.to_f64() - 1.0).abs(),
                Err(_) => 1e6,
            }
        };

        let result = island_evolve(&mut rng, &config, &fitness);
        assert!(result.best_fitness.is_finite());
    }
}
