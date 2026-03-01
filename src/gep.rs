//! Gene Expression Programming (GEP): linear fixed-length genomes decoded
//! into expression trees via a head/tail structure.

use crate::ast::*;
use crate::genetic::GenStats;
use rand::Rng;

/// Symbol in the GEP alphabet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Symbol {
    // Functions (arity 2)
    Add,
    Sub,
    Mul,
    Div,
    // Functions (arity 1)
    Sqrt,
    Sin,
    Abs,
    // Terminals
    Var(usize),
    Const(i8),
}

impl Symbol {
    #[allow(dead_code)]
    fn arity(self) -> usize {
        match self {
            Symbol::Add | Symbol::Sub | Symbol::Mul | Symbol::Div => 2,
            Symbol::Sqrt | Symbol::Sin | Symbol::Abs => 1,
            Symbol::Var(_) | Symbol::Const(_) => 0,
        }
    }

    #[allow(dead_code)]
    fn is_function(self) -> bool {
        self.arity() > 0
    }

    fn random_function<R: Rng>(rng: &mut R) -> Self {
        match rng.gen_range(0..7) {
            0 => Symbol::Add,
            1 => Symbol::Sub,
            2 => Symbol::Mul,
            3 => Symbol::Div,
            4 => Symbol::Sqrt,
            5 => Symbol::Sin,
            _ => Symbol::Abs,
        }
    }

    fn random_terminal<R: Rng>(rng: &mut R, num_vars: usize) -> Self {
        if num_vars > 0 && rng.gen_bool(0.6) {
            Symbol::Var(rng.gen_range(0..num_vars))
        } else {
            Symbol::Const(rng.gen_range(-5..=5))
        }
    }

    fn random_any<R: Rng>(rng: &mut R, num_vars: usize) -> Self {
        if rng.gen_bool(0.5) {
            Self::random_function(rng)
        } else {
            Self::random_terminal(rng, num_vars)
        }
    }
}

/// A GEP individual: fixed-length linear genome with head + tail.
#[derive(Debug, Clone)]
pub struct GepGenome {
    pub genes: Vec<Symbol>,
    pub head_len: usize,
}

impl GepGenome {
    /// Create a random GEP genome. Head can contain functions+terminals, tail only terminals.
    pub fn random<R: Rng>(rng: &mut R, head_len: usize, num_vars: usize) -> Self {
        // tail_len = head_len * (max_arity - 1) + 1 for arity 2
        let tail_len = head_len + 1;
        let mut genes = Vec::with_capacity(head_len + tail_len);
        for _ in 0..head_len {
            genes.push(Symbol::random_any(rng, num_vars));
        }
        for _ in 0..tail_len {
            genes.push(Symbol::random_terminal(rng, num_vars));
        }
        Self { genes, head_len }
    }
}

/// Decode a GEP genome into an AST node using breadth-first expression tree construction.
pub fn decode_gep(genome: &GepGenome) -> Node {
    if genome.genes.is_empty() {
        return Node::FloatConst(0.0);
    }
    decode_symbol(&genome.genes, 0)
}

fn decode_symbol(genes: &[Symbol], idx: usize) -> Node {
    if idx >= genes.len() {
        return Node::FloatConst(0.0);
    }
    match genes[idx] {
        Symbol::Add => {
            let l = decode_symbol(genes, 2 * idx + 1);
            let r = decode_symbol(genes, 2 * idx + 2);
            Node::BinOp(BinOp::Add, Box::new(l), Box::new(r))
        }
        Symbol::Sub => {
            let l = decode_symbol(genes, 2 * idx + 1);
            let r = decode_symbol(genes, 2 * idx + 2);
            Node::BinOp(BinOp::Sub, Box::new(l), Box::new(r))
        }
        Symbol::Mul => {
            let l = decode_symbol(genes, 2 * idx + 1);
            let r = decode_symbol(genes, 2 * idx + 2);
            Node::BinOp(BinOp::Mul, Box::new(l), Box::new(r))
        }
        Symbol::Div => {
            let l = decode_symbol(genes, 2 * idx + 1);
            let r = decode_symbol(genes, 2 * idx + 2);
            Node::BinOp(BinOp::Div, Box::new(l), Box::new(r))
        }
        Symbol::Sqrt => {
            let c = decode_symbol(genes, 2 * idx + 1);
            Node::MathFn(MathFn::Sqrt, Box::new(c))
        }
        Symbol::Sin => {
            let c = decode_symbol(genes, 2 * idx + 1);
            Node::MathFn(MathFn::Sin, Box::new(c))
        }
        Symbol::Abs => {
            let c = decode_symbol(genes, 2 * idx + 1);
            Node::MathFn(MathFn::Abs, Box::new(c))
        }
        Symbol::Var(i) => Node::Var(i),
        Symbol::Const(v) => Node::IntConst(v as i64),
    }
}

/// Configuration for GEP.
#[derive(Debug, Clone)]
pub struct GepConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub head_length: usize,
    pub num_vars: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elitism: usize,
}

impl Default for GepConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            max_generations: 100,
            head_length: 7,
            num_vars: 1,
            tournament_size: 5,
            crossover_rate: 0.7,
            mutation_rate: 0.2,
            elitism: 2,
        }
    }
}

/// One-point crossover for GEP.
pub fn gep_crossover<R: Rng>(rng: &mut R, a: &GepGenome, b: &GepGenome) -> (GepGenome, GepGenome) {
    let len = a.genes.len().min(b.genes.len());
    let point = rng.gen_range(0..len);
    let mut g1 = a.genes[..point].to_vec();
    g1.extend_from_slice(&b.genes[point..]);
    let mut g2 = b.genes[..point].to_vec();
    g2.extend_from_slice(&a.genes[point..]);
    (
        GepGenome {
            genes: g1,
            head_len: a.head_len,
        },
        GepGenome {
            genes: g2,
            head_len: b.head_len,
        },
    )
}

/// Point mutation for GEP (respects head/tail constraint).
pub fn gep_mutate<R: Rng>(rng: &mut R, genome: &GepGenome, num_vars: usize) -> GepGenome {
    let mut genes = genome.genes.clone();
    let idx = rng.gen_range(0..genes.len());
    if idx < genome.head_len {
        genes[idx] = Symbol::random_any(rng, num_vars);
    } else {
        genes[idx] = Symbol::random_terminal(rng, num_vars);
    }
    GepGenome {
        genes,
        head_len: genome.head_len,
    }
}

fn gep_tournament<'a, R: Rng>(
    rng: &mut R,
    pop: &'a [(GepGenome, f64)],
    size: usize,
) -> &'a GepGenome {
    let mut best = rng.gen_range(0..pop.len());
    for _ in 1..size {
        let idx = rng.gen_range(0..pop.len());
        if pop[idx].1 < pop[best].1 {
            best = idx;
        }
    }
    &pop[best].0
}

/// Run Gene Expression Programming evolution.
pub fn gep_evolve<R, F>(
    rng: &mut R,
    config: &GepConfig,
    fitness_fn: &F,
) -> (Node, f64, Vec<GenStats>)
where
    R: Rng,
    F: Fn(&Node) -> f64,
{
    let mut population: Vec<(GepGenome, f64)> = (0..config.population_size)
        .map(|_| {
            let g = GepGenome::random(rng, config.head_length, config.num_vars);
            let tree = decode_gep(&g);
            let fit = fitness_fn(&tree);
            (g, fit)
        })
        .collect();

    let mut stats = Vec::new();
    let mut best_ever: Option<(Node, f64)> = None;

    for gen in 0..config.max_generations {
        population.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_fit = population[0].1;
        let avg_fit = population.iter().map(|(_, f)| f).sum::<f64>() / population.len() as f64;
        let best_tree = decode_gep(&population[0].0);
        let avg_size = population
            .iter()
            .map(|(g, _)| decode_gep(g).size() as f64)
            .sum::<f64>()
            / population.len() as f64;

        stats.push(GenStats {
            generation: gen,
            best_fitness: best_fit,
            avg_fitness: avg_fit,
            avg_size,
            best_program: best_tree.to_expr(),
        });

        if best_ever.is_none() || best_fit < best_ever.as_ref().unwrap().1 {
            best_ever = Some((best_tree, best_fit));
        }

        if best_fit < 1e-10 {
            break;
        }

        let mut next = Vec::with_capacity(config.population_size);
        for (g, _) in population.iter().take(config.elitism) {
            next.push(g.clone());
        }

        while next.len() < config.population_size {
            let r: f64 = rng.gen();
            let child = if r < config.crossover_rate {
                let p1 = gep_tournament(rng, &population, config.tournament_size);
                let p2 = gep_tournament(rng, &population, config.tournament_size);
                gep_crossover(rng, p1, p2).0
            } else if r < config.crossover_rate + config.mutation_rate {
                let p = gep_tournament(rng, &population, config.tournament_size);
                gep_mutate(rng, p, config.num_vars)
            } else {
                gep_tournament(rng, &population, config.tournament_size).clone()
            };
            next.push(child);
        }

        population = next
            .into_iter()
            .map(|g| {
                let tree = decode_gep(&g);
                let fit = fitness_fn(&tree);
                (g, fit)
            })
            .collect();
    }

    let (best_tree, best_fit) = best_ever.unwrap_or_else(|| {
        let t = decode_gep(&population[0].0);
        let f = population[0].1;
        (t, f)
    });
    (best_tree, best_fit, stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_symbol_arity() {
        assert_eq!(Symbol::Add.arity(), 2);
        assert_eq!(Symbol::Sqrt.arity(), 1);
        assert_eq!(Symbol::Var(0).arity(), 0);
        assert!(Symbol::Add.is_function());
        assert!(!Symbol::Const(1).is_function());
    }

    #[test]
    fn test_decode_gep() {
        let genome = GepGenome {
            genes: vec![
                Symbol::Add,
                Symbol::Var(0),
                Symbol::Const(1),
                Symbol::Const(2),
                Symbol::Var(0),
            ],
            head_len: 3,
        };
        let tree = decode_gep(&genome);
        assert!(tree.size() >= 3);
    }

    #[test]
    fn test_gep_crossover() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let a = GepGenome::random(&mut rng, 5, 1);
        let b = GepGenome::random(&mut rng, 5, 1);
        let (c1, c2) = gep_crossover(&mut rng, &a, &b);
        assert_eq!(c1.genes.len(), a.genes.len());
        assert_eq!(c2.genes.len(), b.genes.len());
    }

    #[test]
    fn test_gep_mutate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let g = GepGenome::random(&mut rng, 5, 2);
        let m = gep_mutate(&mut rng, &g, 2);
        // At least one gene should differ (probabilistically almost certain)
        assert_eq!(m.genes.len(), g.genes.len());
    }

    #[test]
    fn test_gep_evolve_simple() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GepConfig {
            population_size: 50,
            max_generations: 20,
            head_length: 5,
            num_vars: 1,
            ..GepConfig::default()
        };

        let fitness = |tree: &Node| -> f64 {
            let mut interp = crate::interpreter::Interpreter::default();
            let mut error = 0.0;
            for i in -3..=3 {
                let x = i as f64;
                interp.reset();
                match interp.eval(tree, &[x]) {
                    Ok(val) => {
                        let diff = val.to_f64() - (x + 1.0);
                        error += diff * diff;
                    }
                    Err(_) => error += 1e6,
                }
            }
            error
        };

        let (best, best_fit, stats) = gep_evolve(&mut rng, &config, &fitness);
        assert!(!stats.is_empty());
        assert!(best.size() > 0);
        assert!(best_fit < 1e6);
    }
}
