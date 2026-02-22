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
        // Should get reasonably close to the target
        assert!(best_fit < 100.0, "fitness too high: {best_fit}");
    }

    #[test]
    fn test_generate_data() {
        let data = generate_data(|x| x * x, -3..=3);
        assert_eq!(data.len(), 7);
        assert_eq!(data[3], (0.0, 0.0));
        assert_eq!(data[6], (3.0, 9.0));
    }
}
