//! Grammatical Evolution (GE): evolves integer codon sequences that are
//! mapped through a BNF-style grammar to produce AST nodes.

use crate::ast::*;
use crate::genetic::GenStats;
use rand::Rng;

/// A GE individual: a sequence of integer codons.
#[derive(Debug, Clone)]
pub struct GeGenome {
    pub codons: Vec<u32>,
}

impl GeGenome {
    pub fn random<R: Rng>(rng: &mut R, length: usize) -> Self {
        Self {
            codons: (0..length).map(|_| rng.gen()).collect(),
        }
    }
}

/// Configuration for Grammatical Evolution.
#[derive(Debug, Clone)]
pub struct GeConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub codon_length: usize,
    pub max_wraps: usize,
    pub num_vars: usize,
    pub tournament_size: usize,
    pub crossover_rate: f64,
    pub mutation_rate: f64,
    pub elitism: usize,
    pub max_depth: usize,
}

impl Default for GeConfig {
    fn default() -> Self {
        Self {
            population_size: 200,
            max_generations: 100,
            codon_length: 100,
            max_wraps: 3,
            num_vars: 1,
            tournament_size: 5,
            crossover_rate: 0.7,
            mutation_rate: 0.2,
            elitism: 2,
            max_depth: 8,
        }
    }
}

/// Decoder state: walks through codons producing AST nodes.
struct Decoder<'a> {
    codons: &'a [u32],
    pos: usize,
    wraps: usize,
    max_wraps: usize,
    num_vars: usize,
    max_depth: usize,
    exhausted: bool,
}

impl<'a> Decoder<'a> {
    fn new(codons: &'a [u32], max_wraps: usize, num_vars: usize, max_depth: usize) -> Self {
        Self {
            codons,
            pos: 0,
            wraps: 0,
            max_wraps,
            num_vars,
            max_depth,
            exhausted: false,
        }
    }

    fn next_codon(&mut self) -> u32 {
        if self.exhausted || self.codons.is_empty() {
            self.exhausted = true;
            return 0;
        }
        let val = self.codons[self.pos];
        self.pos += 1;
        if self.pos >= self.codons.len() {
            self.wraps += 1;
            self.pos = 0;
            if self.wraps > self.max_wraps {
                self.exhausted = true;
            }
        }
        val
    }

    fn decode(&mut self, depth: usize) -> Node {
        if self.exhausted || depth >= self.max_depth {
            return self.decode_terminal();
        }

        let codon = self.next_codon();
        match codon as usize % 8 {
            0 => {
                let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div];
                let op = ops[self.next_codon() as usize % 4];
                let l = self.decode(depth + 1);
                let r = self.decode(depth + 1);
                Node::BinOp(op, Box::new(l), Box::new(r))
            }
            1 => {
                let cond = self.decode(depth + 1);
                let then = self.decode(depth + 1);
                let els = self.decode(depth + 1);
                Node::If(Box::new(cond), Box::new(then), Box::new(els))
            }
            2 => {
                let fns = [
                    MathFn::Abs,
                    MathFn::Sqrt,
                    MathFn::Sin,
                    MathFn::Cos,
                    MathFn::Exp,
                    MathFn::Log,
                ];
                let f = fns[self.next_codon() as usize % 6];
                let c = self.decode(depth + 1);
                Node::MathFn(f, Box::new(c))
            }
            3 => {
                let ops = [CmpOp::Lt, CmpOp::Gt, CmpOp::Eq];
                let op = ops[self.next_codon() as usize % 3];
                let l = self.decode(depth + 1);
                let r = self.decode(depth + 1);
                Node::Cmp(op, Box::new(l), Box::new(r))
            }
            4 => {
                let ops = [UnaryOp::Neg, UnaryOp::Not];
                let op = ops[self.next_codon() as usize % 2];
                let c = self.decode(depth + 1);
                Node::UnaryOp(op, Box::new(c))
            }
            _ => self.decode_terminal(),
        }
    }

    fn decode_terminal(&mut self) -> Node {
        let codon = self.next_codon();
        match codon as usize % 3 {
            0 => {
                if self.num_vars > 0 {
                    Node::Var(codon as usize % self.num_vars)
                } else {
                    Node::FloatConst(1.0)
                }
            }
            1 => Node::FloatConst((codon % 200) as f64 / 10.0 - 10.0),
            _ => Node::IntConst((codon % 21) as i64 - 10),
        }
    }
}

/// Decode a GE genome into an AST node.
pub fn decode_genome(genome: &GeGenome, config: &GeConfig) -> Node {
    let mut decoder = Decoder::new(
        &genome.codons,
        config.max_wraps,
        config.num_vars,
        config.max_depth,
    );
    decoder.decode(0)
}

/// Single-point crossover on codon sequences.
pub fn ge_crossover<R: Rng>(rng: &mut R, a: &GeGenome, b: &GeGenome) -> (GeGenome, GeGenome) {
    let len = a.codons.len().min(b.codons.len());
    let point = rng.gen_range(0..len);
    let mut c1 = a.codons[..point].to_vec();
    c1.extend_from_slice(&b.codons[point..]);
    let mut c2 = b.codons[..point].to_vec();
    c2.extend_from_slice(&a.codons[point..]);
    (GeGenome { codons: c1 }, GeGenome { codons: c2 })
}

/// Point mutation: flip random codons.
pub fn ge_mutate<R: Rng>(rng: &mut R, genome: &GeGenome, rate: f64) -> GeGenome {
    let codons = genome
        .codons
        .iter()
        .map(|&c| {
            if rng.gen::<f64>() < rate {
                rng.gen()
            } else {
                c
            }
        })
        .collect();
    GeGenome { codons }
}

fn ge_tournament<'a, R: Rng>(rng: &mut R, pop: &'a [(GeGenome, f64)], size: usize) -> &'a GeGenome {
    let mut best = rng.gen_range(0..pop.len());
    for _ in 1..size {
        let idx = rng.gen_range(0..pop.len());
        if pop[idx].1 < pop[best].1 {
            best = idx;
        }
    }
    &pop[best].0
}

/// Run Grammatical Evolution.
pub fn ge_evolve<R, F>(rng: &mut R, config: &GeConfig, fitness_fn: &F) -> (Node, f64, Vec<GenStats>)
where
    R: Rng,
    F: Fn(&Node) -> f64,
{
    let mut population: Vec<(GeGenome, f64)> = (0..config.population_size)
        .map(|_| {
            let g = GeGenome::random(rng, config.codon_length);
            let tree = decode_genome(&g, config);
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
        let best_tree = decode_genome(&population[0].0, config);
        let avg_size = population
            .iter()
            .map(|(g, _)| decode_genome(g, config).size() as f64)
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
                let p1 = ge_tournament(rng, &population, config.tournament_size);
                let p2 = ge_tournament(rng, &population, config.tournament_size);
                ge_crossover(rng, p1, p2).0
            } else if r < config.crossover_rate + config.mutation_rate {
                let p = ge_tournament(rng, &population, config.tournament_size);
                ge_mutate(rng, p, 0.05)
            } else {
                ge_tournament(rng, &population, config.tournament_size).clone()
            };
            next.push(child);
        }

        population = next
            .into_iter()
            .map(|g| {
                let tree = decode_genome(&g, config);
                let fit = fitness_fn(&tree);
                (g, fit)
            })
            .collect();
    }

    let (best_tree, best_fit) = best_ever.unwrap_or_else(|| {
        let t = decode_genome(&population[0].0, config);
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
    fn test_decode_deterministic() {
        let genome = GeGenome {
            codons: vec![0, 1, 5, 3, 7, 2, 8, 4, 6, 9],
        };
        let config = GeConfig {
            num_vars: 2,
            ..GeConfig::default()
        };
        let tree = decode_genome(&genome, &config);
        assert!(tree.size() > 0);
    }

    #[test]
    fn test_ge_crossover() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let a = GeGenome::random(&mut rng, 20);
        let b = GeGenome::random(&mut rng, 20);
        let (c1, c2) = ge_crossover(&mut rng, &a, &b);
        assert_eq!(c1.codons.len(), 20);
        assert_eq!(c2.codons.len(), 20);
    }

    #[test]
    fn test_ge_mutate() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let g = GeGenome {
            codons: vec![0; 20],
        };
        let m = ge_mutate(&mut rng, &g, 1.0);
        assert!(m.codons.iter().any(|&c| c != 0));
    }

    #[test]
    fn test_ge_evolve_simple() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let config = GeConfig {
            population_size: 50,
            max_generations: 20,
            codon_length: 50,
            num_vars: 1,
            ..GeConfig::default()
        };

        let fitness = |tree: &Node| -> f64 {
            let mut interp = crate::interpreter::Interpreter::default();
            let mut error = 0.0;
            for i in -3..=3 {
                let x = i as f64;
                interp.reset();
                match interp.eval(tree, &[x]) {
                    Ok(val) => {
                        let diff = val.to_f64() - (2.0 * x + 1.0);
                        error += diff * diff;
                    }
                    Err(_) => error += 1e6,
                }
            }
            error
        };

        let (best, best_fit, stats) = ge_evolve(&mut rng, &config, &fitness);
        assert!(!stats.is_empty());
        assert!(best.size() > 0);
        assert!(best_fit < 1e6);
    }
}
