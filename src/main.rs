use genlang::adf::{adf_evolve, eval_adf_program, AdfConfig};
use genlang::cartpole::{cart_pole, simulate, CartPoleParams};
use genlang::coevolution::{coevolve, CoevConfig, Outcome};
use genlang::genetic::GpConfig;
use genlang::io::{save_genome, save_stats, SavedGenome};
use genlang::island::{island_evolve, IslandConfig};
use genlang::map_elites::{map_elites_evolve, MapElitesConfig};
use genlang::novelty::{novelty_evolve, NoveltyConfig};
use genlang::pareto::{nsga2_evolve, NsgaConfig};
use genlang::problems::{
    even_parity, fibonacci, generate_data, maze_solver, sorting_network, symbolic_regression,
};
use genlang::speciation::{speciated_evolve, SpeciationConfig};
use genlang::visualization::{print_evolution_summary, sparkline, to_mermaid, tree_print};
use rand::SeedableRng;

fn main() {
    println!(
        "🧬 genlang v{} — Self-Evolving Programming Language",
        env!("CARGO_PKG_VERSION")
    );
    println!("====================================================\n");

    demo_symbolic_regression();
    println!();
    demo_even_parity();
    println!();
    demo_fibonacci();
    println!();
    demo_island_model();
    println!();
    demo_multi_objective();
    println!();
    demo_sorting();
    println!();
    demo_maze();
    println!();
    demo_novelty_search();
    println!();
    demo_speciation();
    println!();
    demo_cart_pole();
    println!();
    demo_adfs();
    println!();
    demo_map_elites();
    println!();
    demo_coevolution();
    println!();
    demo_grammatical_evolution();
    println!();
    demo_gene_expression_programming();
    println!();
    demo_convergence_dashboard();
}

fn demo_symbolic_regression() {
    println!("📐 Demo 1: Symbolic Regression (x² + x + 1)");
    println!("---------------------------------------------");

    let data = generate_data(|x| x * x + x + 1.0, -10..=10);
    let config = GpConfig {
        population_size: 500,
        max_generations: 100,
        max_depth: 6,
        num_vars: 1,
        ..GpConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let (best, best_fit, stats) = symbolic_regression(&mut rng, &data, &config);

    for s in &stats {
        if s.generation % 20 == 0 || s.generation == stats.len() - 1 {
            println!(
                "  Gen {:3} | Fit: {:.6} | Size: {:.1} | {}",
                s.generation, s.best_fitness, s.avg_size, s.best_program
            );
        }
    }

    print_evolution_summary(&stats);

    println!("\n  🏆 Best: {} (MSE: {:.6})", best.to_expr(), best_fit);
    println!("\n  🌳 Tree structure:");
    for line in tree_print(&best).lines() {
        println!("    {line}");
    }

    let saved = SavedGenome::new(best)
        .with_fitness(best_fit)
        .with_description("symbolic regression: x² + x + 1");
    if save_genome("best_symreg.json", &saved).is_ok() {
        println!("\n  💾 Genome saved to best_symreg.json");
    }
    let _ = save_stats("symreg_stats.json", &stats);
}

fn demo_even_parity() {
    println!("🔢 Demo 2: Even Parity (2-bit)");
    println!("-------------------------------");

    let config = GpConfig {
        population_size: 300,
        max_generations: 80,
        max_depth: 5,
        num_vars: 2,
        ..GpConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(123);
    let (best, best_fit, stats) = even_parity(&mut rng, 2, &config);

    print_evolution_summary(&stats);
    println!("  🏆 Best: {} (errors: {:.0})", best.to_expr(), best_fit);

    println!("  Verification:");
    let mut interp = genlang::interpreter::Interpreter::default();
    for (a, b) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
        interp.reset();
        let out = interp
            .eval(&best, &[a, b])
            .map(|v| v.to_f64())
            .unwrap_or(f64::NAN);
        let expected = if ((a > 0.5) as u8 + (b > 0.5) as u8).is_multiple_of(2) {
            1
        } else {
            0
        };
        let predicted = if out > 0.5 { 1 } else { 0 };
        let mark = if predicted == expected { "✓" } else { "✗" };
        println!("    ({a}, {b}) → {predicted} (expected {expected}) {mark}");
    }
}

fn demo_fibonacci() {
    println!("🔄 Demo 3: Fibonacci (0..10)");
    println!("-----------------------------");

    let config = GpConfig {
        population_size: 500,
        max_generations: 100,
        max_depth: 6,
        num_vars: 1,
        ..GpConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(55);
    let (best, best_fit, stats) = fibonacci(&mut rng, 10, &config);

    print_evolution_summary(&stats);
    println!("  🏆 Best: {} (MSE: {:.4})", best.to_expr(), best_fit);

    // Verify against actual Fibonacci
    let fibs = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
    println!("  Verification:");
    let mut interp = genlang::interpreter::Interpreter::default();
    for (i, &expected) in fibs.iter().enumerate() {
        interp.reset();
        let out = interp
            .eval(&best, &[i as f64])
            .map(|v| v.to_f64())
            .unwrap_or(f64::NAN);
        let mark = if (out - expected as f64).abs() < 1.0 {
            "✓"
        } else {
            "✗"
        };
        println!("    fib({i}) = {out:.1} (expected {expected}) {mark}");
    }
}

fn demo_island_model() {
    println!("🏝️  Demo 4: Island Model (target: 3x + 2)");
    println!("------------------------------------------");

    let config = IslandConfig {
        num_islands: 4,
        migration_interval: 15,
        migration_size: 3,
        gp_config: GpConfig {
            population_size: 100,
            max_depth: 5,
            num_vars: 1,
            ..GpConfig::default()
        },
        total_generations: 60,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(77);

    let fitness = |tree: &genlang::ast::Node| -> f64 {
        let mut interp = genlang::interpreter::Interpreter::default();
        let mut error = 0.0;
        for i in -5..=5 {
            let x = i as f64;
            interp.reset();
            match interp.eval(tree, &[x]) {
                Ok(val) => {
                    let diff = val.to_f64() - (3.0 * x + 2.0);
                    error += diff * diff;
                }
                Err(_) => error += 1e6,
            }
        }
        error / 11.0
    };

    let result = island_evolve(&mut rng, &config, &fitness);
    println!(
        "  Islands: {} | Migrations every {} gens",
        config.num_islands, config.migration_interval
    );
    println!(
        "  🏆 Best: {} (MSE: {:.6})",
        result.best_genome.to_expr(),
        result.best_fitness
    );

    println!("\n  📊 Mermaid diagram:");
    println!("  ```mermaid");
    for line in to_mermaid(&result.best_genome).lines() {
        println!("  {line}");
    }
    println!("  ```");
}

fn demo_multi_objective() {
    println!("⚖️  Demo 5: Multi-Objective Pareto (accuracy vs simplicity)");
    println!("------------------------------------------------------------");

    let config = NsgaConfig {
        population_size: 200,
        max_generations: 50,
        max_depth: 5,
        num_vars: 1,
        ..NsgaConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Objective 1: accuracy on x^2
    let obj_accuracy = |tree: &genlang::ast::Node| -> f64 {
        let mut interp = genlang::interpreter::Interpreter::default();
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
        error / 11.0
    };

    // Objective 2: program size (simplicity)
    let obj_simplicity = |tree: &genlang::ast::Node| -> f64 { tree.size() as f64 };

    let (pareto_front, stats) = nsga2_evolve(&mut rng, &config, &[&obj_accuracy, &obj_simplicity]);

    // Show convergence
    let front_sizes: Vec<f64> = stats.iter().map(|s| s.pareto_front_size as f64).collect();
    println!("  Pareto front size: {}", sparkline(&front_sizes));

    println!("  Final Pareto front: {} solutions", pareto_front.len());

    // Show top 5 solutions sorted by accuracy
    let mut sorted = pareto_front.clone();
    sorted.sort_by(|a, b| {
        a.objectives[0]
            .partial_cmp(&b.objectives[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("  Top solutions (accuracy → simplicity tradeoff):");
    for (i, sol) in sorted.iter().take(5).enumerate() {
        println!(
            "    #{}: MSE={:.4}, size={:.0}, expr={}",
            i + 1,
            sol.objectives[0],
            sol.objectives[1],
            sol.genome.to_expr()
        );
    }
}

fn demo_sorting() {
    println!("🔀 Demo 6: Sorting Network Evolution (4 elements)");
    println!("--------------------------------------------------");

    let config = GpConfig {
        population_size: 300,
        max_generations: 60,
        max_depth: 4,
        num_vars: 2,
        ..GpConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let (best, best_fit, stats) = sorting_network(&mut rng, 4, &config);

    print_evolution_summary(&stats);
    println!(
        "  🏆 Best comparator: {} (inversions: {:.0})",
        best.to_expr(),
        best_fit
    );
}

fn demo_maze() {
    println!("🧭 Demo 7: Maze Solver Evolution");
    println!("----------------------------------");

    let config = GpConfig {
        population_size: 300,
        max_generations: 60,
        max_depth: 5,
        num_vars: 4,
        ..GpConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let (best, best_fit, stats) = maze_solver(&mut rng, &config);

    print_evolution_summary(&stats);
    println!("  🏆 Best: {} (errors: {:.0}/11)", best.to_expr(), best_fit);
}

fn demo_novelty_search() {
    println!("🔍 Demo 8: Novelty Search");
    println!("-------------------------");

    let config = NoveltyConfig {
        gp: GpConfig {
            population_size: 100,
            max_generations: 30,
            max_depth: 4,
            num_vars: 1,
            ..GpConfig::default()
        },
        k_nearest: 10,
        fitness_weight: 0.0,
        ..NoveltyConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let behavior_fn = |tree: &genlang::ast::Node| -> Vec<f64> {
        let mut interp = genlang::interpreter::Interpreter::default();
        (-3..=3)
            .map(|i| {
                interp.reset();
                interp
                    .eval(tree, &[i as f64])
                    .map(|v| v.to_f64().clamp(-100.0, 100.0))
                    .unwrap_or(0.0)
            })
            .collect()
    };

    let fitness_fn = |_: &genlang::ast::Node| -> f64 { 0.0 };

    let (archive, stats) = novelty_evolve(&mut rng, &config, &behavior_fn, &fitness_fn);

    let novelties: Vec<f64> = stats.iter().map(|s| s.best_fitness).collect();
    println!("  Novelty: {}", sparkline(&novelties));
    println!("  Archive size: {} unique behaviors", archive.len());
    if let Some(most_novel) = archive.iter().max_by(|a, b| {
        a.novelty_score
            .partial_cmp(&b.novelty_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "  Most novel: {} (score: {:.2})",
            most_novel.genome.to_expr(),
            most_novel.novelty_score
        );
    }
}

fn demo_speciation() {
    println!("🧬 Demo 9: Speciated Evolution (NEAT-inspired)");
    println!("-----------------------------------------------");

    let config = SpeciationConfig {
        gp: GpConfig {
            population_size: 200,
            max_generations: 50,
            max_depth: 5,
            num_vars: 1,
            ..GpConfig::default()
        },
        compatibility_threshold: 5.0,
        max_stagnation: 15,
        min_species_size: 5,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Target: x^2 - 3x + 2
    let fitness = |tree: &genlang::ast::Node| -> f64 {
        let mut interp = genlang::interpreter::Interpreter::default();
        let mut error = 0.0;
        for i in -5..=5 {
            let x = i as f64;
            interp.reset();
            match interp.eval(tree, &[x]) {
                Ok(val) => {
                    let diff = val.to_f64() - (x * x - 3.0 * x + 2.0);
                    error += diff * diff;
                }
                Err(_) => error += 1e6,
            }
        }
        error / 11.0
    };

    let (best, best_fit, stats) = speciated_evolve(&mut rng, &config, &fitness);

    print_evolution_summary(&stats);
    println!("  🏆 Best: {} (MSE: {:.6})", best.to_expr(), best_fit);
    println!("\n  🌳 Tree:");
    for line in tree_print(&best).lines() {
        println!("    {line}");
    }
}

fn demo_cart_pole() {
    println!("⚖️  Demo 10: Cart-Pole Balancing");
    println!("---------------------------------");

    let config = GpConfig {
        population_size: 300,
        max_generations: 60,
        max_depth: 5,
        num_vars: 4, // x, x_dot, theta, theta_dot
        ..GpConfig::default()
    };
    let params = CartPoleParams {
        max_steps: 500,
        ..CartPoleParams::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let (best, best_fit, stats) = cart_pole(&mut rng, &config, &params);

    print_evolution_summary(&stats);
    let steps = simulate(&best, &params);
    println!(
        "  🏆 Best controller: {} (survived: {} steps, fitness: {:.1})",
        best.to_expr(),
        steps,
        best_fit
    );
    println!("  (max possible: {} steps)", params.max_steps);
}

fn demo_adfs() {
    println!("🧩 Demo 11: Automatically Defined Functions (ADFs)");
    println!("---------------------------------------------------");

    let config = AdfConfig {
        gp: GpConfig {
            population_size: 200,
            max_generations: 50,
            max_depth: 5,
            num_vars: 1,
            ..GpConfig::default()
        },
        num_adfs: 2,
        adf_arity: 2,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Target: x^3 + x^2 + x (benefits from reusable x^2 subroutine)
    let fitness = |prog: &genlang::adf::AdfProgram| -> f64 {
        let mut interp = genlang::interpreter::Interpreter::default();
        let mut error = 0.0;
        for i in -5..=5 {
            let x = i as f64;
            interp.reset();
            match eval_adf_program(&mut interp, prog, &[x]) {
                Ok(val) => {
                    let diff = val.to_f64() - (x * x * x + x * x + x);
                    error += diff * diff;
                }
                Err(_) => error += 1e6,
            }
        }
        error / 11.0
    };

    let (best, best_fit, stats) = adf_evolve(&mut rng, &config, &fitness);

    print_evolution_summary(&stats);
    println!("  🏆 Best (MSE: {:.4}):", best_fit);
    for line in best.to_string_pretty().lines() {
        println!("    {line}");
    }
}

fn demo_map_elites() {
    println!("🗺️  Demo 12: MAP-Elites (Quality-Diversity)");
    println!("--------------------------------------------");

    let config = MapElitesConfig {
        iterations: 3000,
        initial_pop: 100,
        max_depth: 5,
        max_tree_size: 60,
        num_vars: 1,
        bins_per_dim: 8,
        behavior_ranges: vec![(-10.0, 10.0), (0.0, 40.0)],
        mutation_rate: 0.7,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Fitness: accuracy on x^2
    let fitness_fn = |tree: &genlang::ast::Node| -> f64 {
        let mut interp = genlang::interpreter::Interpreter::default();
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
        error / 11.0
    };

    // Behavior: (output at x=0, program size)
    let behavior_fn = |tree: &genlang::ast::Node| -> Vec<f64> {
        let mut interp = genlang::interpreter::Interpreter::default();
        let out = interp
            .eval(tree, &[0.0])
            .map(|v| v.to_f64().clamp(-10.0, 10.0))
            .unwrap_or(0.0);
        vec![out, tree.size() as f64]
    };

    let (archive, stats) = map_elites_evolve(&mut rng, &config, &fitness_fn, &behavior_fn);

    let coverages: Vec<f64> = stats.iter().map(|s| s.coverage * 100.0).collect();
    println!("  Coverage: {}", sparkline(&coverages));
    println!(
        "  Filled cells: {} / {} ({:.1}%)",
        archive.filled(),
        config.bins_per_dim * config.bins_per_dim,
        archive.coverage() * 100.0
    );
    if let Some(best) = archive.best() {
        println!(
            "  🏆 Best: {} (fitness: {:.4})",
            best.genome.to_expr(),
            best.fitness
        );
    }
}

fn demo_coevolution() {
    println!("⚔️  Demo 13: Competitive Coevolution");
    println!("-------------------------------------");

    let config = CoevConfig {
        config_a: GpConfig {
            population_size: 80,
            max_generations: 30,
            max_depth: 4,
            num_vars: 1,
            ..GpConfig::default()
        },
        config_b: GpConfig {
            population_size: 80,
            max_generations: 30,
            max_depth: 4,
            num_vars: 1,
            ..GpConfig::default()
        },
        sample_opponents: 8,
        generations: 30,
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Number war: each program outputs a number given input 1.0, higher wins
    let compete = |a: &genlang::ast::Node, b: &genlang::ast::Node| -> Outcome {
        let mut interp = genlang::interpreter::Interpreter::default();
        let va = interp.eval(a, &[1.0]).map(|v| v.to_f64()).unwrap_or(0.0);
        interp.reset();
        let vb = interp.eval(b, &[1.0]).map(|v| v.to_f64()).unwrap_or(0.0);
        if va > vb + 0.01 {
            Outcome::Win
        } else if vb > va + 0.01 {
            Outcome::Loss
        } else {
            Outcome::Draw
        }
    };

    let (best_a, best_b, stats) = coevolve(&mut rng, &config, &compete);

    let fit_a: Vec<f64> = stats.iter().map(|s| -s.best_fitness_a).collect();
    let fit_b: Vec<f64> = stats.iter().map(|s| -s.best_fitness_b).collect();
    println!("  Pop A win rate: {}", sparkline(&fit_a));
    println!("  Pop B win rate: {}", sparkline(&fit_b));
    println!("  🏆 Best A: {}", best_a.to_expr());
    println!("  🏆 Best B: {}", best_b.to_expr());
}

fn demo_grammatical_evolution() {
    println!("📝 Demo 14: Grammatical Evolution (GE)");
    println!("---------------------------------------");

    use genlang::ge::{ge_evolve, GeConfig};

    let config = GeConfig {
        population_size: 300,
        max_generations: 60,
        codon_length: 80,
        num_vars: 1,
        ..GeConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Target: 2x + 3
    let fitness = |tree: &genlang::ast::Node| -> f64 {
        let mut interp = genlang::interpreter::Interpreter::default();
        let mut error = 0.0;
        for i in -5..=5 {
            let x = i as f64;
            interp.reset();
            match interp.eval(tree, &[x]) {
                Ok(val) => {
                    let diff = val.to_f64() - (2.0 * x + 3.0);
                    error += diff * diff;
                }
                Err(_) => error += 1e6,
            }
        }
        error / 11.0
    };

    let (best, best_fit, stats) = ge_evolve(&mut rng, &config, &fitness);

    print_evolution_summary(&stats);
    println!("  🏆 Best: {} (MSE: {:.6})", best.to_expr(), best_fit);
}

fn demo_gene_expression_programming() {
    println!("🧬 Demo 15: Gene Expression Programming (GEP)");
    println!("-----------------------------------------------");

    use genlang::gep::{gep_evolve, GepConfig};

    let config = GepConfig {
        population_size: 300,
        max_generations: 60,
        head_length: 7,
        num_vars: 1,
        ..GepConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // Target: x^2 - 1
    let fitness = |tree: &genlang::ast::Node| -> f64 {
        let mut interp = genlang::interpreter::Interpreter::default();
        let mut error = 0.0;
        for i in -5..=5 {
            let x = i as f64;
            interp.reset();
            match interp.eval(tree, &[x]) {
                Ok(val) => {
                    let diff = val.to_f64() - (x * x - 1.0);
                    error += diff * diff;
                }
                Err(_) => error += 1e6,
            }
        }
        error / 11.0
    };

    let (best, best_fit, stats) = gep_evolve(&mut rng, &config, &fitness);

    print_evolution_summary(&stats);
    println!("  🏆 Best: {} (MSE: {:.6})", best.to_expr(), best_fit);
}

fn demo_convergence_dashboard() {
    println!("📊 Demo 16: Convergence Dashboard");
    println!("----------------------------------");

    use genlang::dashboard::save_dashboard;

    // Run a quick evolution to get stats
    let config = GpConfig {
        population_size: 200,
        max_generations: 50,
        max_depth: 5,
        num_vars: 1,
        ..GpConfig::default()
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let data = generate_data(|x| x * x + 1.0, -5..=5);
    let (_, _, stats) = symbolic_regression(&mut rng, &data, &config);

    match save_dashboard("dashboard.html", &stats, "x² + 1 Regression") {
        Ok(()) => println!("  💾 Dashboard saved to dashboard.html"),
        Err(e) => println!("  ⚠️  Could not save dashboard: {e}"),
    }
    println!("  📈 {} generations tracked", stats.len());
}
