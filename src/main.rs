use genlang::genetic::GpConfig;
use genlang::io::{save_genome, save_stats, SavedGenome};
use genlang::island::{island_evolve, IslandConfig};
use genlang::pareto::{nsga2_evolve, NsgaConfig};
use genlang::problems::{even_parity, fibonacci, generate_data, symbolic_regression};
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
