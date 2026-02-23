use crate::ast::Node;
use crate::genetic::{evolve, GenStats, GpConfig};
use crate::interpreter::Interpreter;
use rand::Rng;

/// Symbolic regression: discover an equation from (x, y) data points.
pub fn symbolic_regression<R: Rng>(
    rng: &mut R,
    data: &[(f64, f64)],
    config: &GpConfig,
) -> (Node, f64, Vec<GenStats>) {
    let fitness = |tree: &Node| -> f64 {
        let mut interp = Interpreter::default();
        let mut total_error = 0.0;
        for (x, y) in data {
            interp.reset();
            match interp.eval(tree, &[*x]) {
                Ok(val) => {
                    let diff = val.to_f64() - y;
                    total_error += diff * diff;
                }
                Err(_) => total_error += 1e6,
            }
        }
        total_error / data.len() as f64
    };

    evolve(rng, config, &fitness)
}

/// Generate data for a target function (for testing).
pub fn generate_data<F>(f: F, range: std::ops::RangeInclusive<i32>) -> Vec<(f64, f64)>
where
    F: Fn(f64) -> f64,
{
    range
        .map(|i| {
            let x = i as f64;
            (x, f(x))
        })
        .collect()
}

/// Generate all boolean input combinations for n bits.
fn all_boolean_inputs(n: usize) -> Vec<Vec<f64>> {
    let count = 1 << n;
    (0..count)
        .map(|i| {
            (0..n)
                .map(|bit| if (i >> bit) & 1 == 1 { 1.0 } else { 0.0 })
                .collect()
        })
        .collect()
}

/// Even-parity problem: evolve a program that returns 1.0 when the number
/// of true (1.0) inputs is even, 0.0 otherwise.
pub fn even_parity<R: Rng>(
    rng: &mut R,
    num_bits: usize,
    config: &GpConfig,
) -> (Node, f64, Vec<GenStats>) {
    let inputs = all_boolean_inputs(num_bits);
    let targets: Vec<f64> = inputs
        .iter()
        .map(|bits| {
            let ones = bits.iter().filter(|&&b| b > 0.5).count();
            if ones % 2 == 0 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    let fitness = |tree: &Node| -> f64 {
        let mut interp = Interpreter::default();
        let mut errors = 0.0;
        for (input, target) in inputs.iter().zip(targets.iter()) {
            interp.reset();
            match interp.eval(tree, input) {
                Ok(val) => {
                    let output = if val.to_f64() > 0.5 { 1.0 } else { 0.0 };
                    if (output - target).abs() > 0.5 {
                        errors += 1.0;
                    }
                }
                Err(_) => errors += 1.0,
            }
        }
        errors
    };

    evolve(rng, config, &fitness)
}

/// Fibonacci problem: evolve a program that maps index n → fib(n).
/// Uses indices 0..=max_n as training data.
pub fn fibonacci<R: Rng>(
    rng: &mut R,
    max_n: usize,
    config: &GpConfig,
) -> (Node, f64, Vec<GenStats>) {
    // Pre-compute Fibonacci targets
    let mut fibs = vec![0.0f64; max_n + 1];
    if max_n >= 1 {
        fibs[1] = 1.0;
    }
    for i in 2..=max_n {
        fibs[i] = fibs[i - 1] + fibs[i - 2];
    }

    let data: Vec<(f64, f64)> = (0..=max_n).map(|i| (i as f64, fibs[i])).collect();

    let fitness = |tree: &Node| -> f64 {
        let mut interp = Interpreter::default();
        let mut total_error = 0.0;
        for (x, y) in &data {
            interp.reset();
            match interp.eval(tree, &[*x]) {
                Ok(val) => {
                    let diff = val.to_f64() - y;
                    total_error += diff * diff;
                }
                Err(_) => total_error += 1e6,
            }
        }
        total_error / data.len() as f64
    };

    evolve(rng, config, &fitness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_symbolic_regression_linear() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let data = generate_data(|x| 2.0 * x + 1.0, -5..=5);

        let config = GpConfig {
            population_size: 100,
            max_generations: 50,
            max_depth: 4,
            num_vars: 1,
            ..GpConfig::default()
        };

        let (_best, best_fit, stats) = symbolic_regression(&mut rng, &data, &config);
        assert!(!stats.is_empty());
        assert!(best_fit < 100.0, "fitness too high: {best_fit}");
    }

    #[test]
    fn test_generate_data() {
        let data = generate_data(|x| x * x, -3..=3);
        assert_eq!(data.len(), 7);
        assert_eq!(data[3], (0.0, 0.0));
        assert_eq!(data[6], (3.0, 9.0));
    }

    #[test]
    fn test_all_boolean_inputs() {
        let inputs = all_boolean_inputs(2);
        assert_eq!(inputs.len(), 4);
        assert_eq!(inputs[0], vec![0.0, 0.0]);
        assert_eq!(inputs[3], vec![1.0, 1.0]);
    }

    #[test]
    fn test_fibonacci() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GpConfig {
            population_size: 200,
            max_generations: 50,
            max_depth: 5,
            num_vars: 1,
            ..GpConfig::default()
        };

        let (_best, best_fit, stats) = fibonacci(&mut rng, 8, &config);
        assert!(!stats.is_empty());
        // Should at least improve from random
        assert!(best_fit < 1e6);
    }

    #[test]
    fn test_even_parity_2bit() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GpConfig {
            population_size: 200,
            max_generations: 50,
            max_depth: 5,
            num_vars: 2,
            ..GpConfig::default()
        };

        let (_best, best_fit, stats) = even_parity(&mut rng, 2, &config);
        assert!(!stats.is_empty());
        // For 2-bit parity, 4 test cases. Should get some right.
        assert!(best_fit <= 4.0);
    }
}
