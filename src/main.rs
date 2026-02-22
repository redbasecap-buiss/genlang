use genlang::genetic::GpConfig;
use genlang::problems::{generate_data, symbolic_regression};
use rand::SeedableRng;

fn main() {
    println!("🧬 genlang — Self-Evolving Programming Language");
    println!("================================================\n");

    // Demo: symbolic regression on x^2 + x + 1
    let data = generate_data(|x| x * x + x + 1.0, -10..=10);

    let config = GpConfig {
        population_size: 500,
        max_generations: 100,
        max_depth: 6,
        num_vars: 1,
        ..GpConfig::default()
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    println!("Target function: x² + x + 1");
    println!("Training on {} data points\n", data.len());
    println!(
        "Config: pop={}, gens={}, depth={}",
        config.population_size, config.max_generations, config.max_depth
    );
    println!("---");

    let (best, best_fit, stats) = symbolic_regression(&mut rng, &data, &config);

    // Print evolution progress
    for s in &stats {
        if s.generation % 10 == 0 || s.generation == stats.len() - 1 {
            println!(
                "Gen {:3} | Best: {:.6} | Avg: {:.2} | Size: {:.1} | {}",
                s.generation, s.best_fitness, s.avg_fitness, s.avg_size, s.best_program
            );
        }
    }

    println!("\n🏆 Best program: {}", best.to_expr());
    println!("   Fitness (MSE): {:.6}", best_fit);
    println!("   Tree size: {} nodes", best.size());

    // Verify on a few points
    println!("\nVerification:");
    let mut interp = genlang::interpreter::Interpreter::default();
    for x in [-2.0, 0.0, 1.0, 3.0, 5.0] {
        interp.reset();
        let predicted = interp
            .eval(&best, &[x])
            .map(|v| v.to_f64())
            .unwrap_or(f64::NAN);
        let expected = x * x + x + 1.0;
        println!("  x={x:5.1} | predicted={predicted:10.4} | expected={expected:10.4}");
    }
}
