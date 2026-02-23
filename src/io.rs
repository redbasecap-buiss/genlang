use crate::ast::Node;
use crate::genetic::GenStats;
use std::fs;
use std::io;
use std::path::Path;

/// Saved genome with metadata.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct SavedGenome {
    pub version: String,
    pub fitness: Option<f64>,
    pub generation: Option<usize>,
    pub description: Option<String>,
    pub genome: Node,
}

impl SavedGenome {
    pub fn new(genome: Node) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            fitness: None,
            generation: None,
            description: None,
            genome,
        }
    }

    pub fn with_fitness(mut self, fitness: f64) -> Self {
        self.fitness = Some(fitness);
        self
    }

    pub fn with_generation(mut self, gen: usize) -> Self {
        self.generation = Some(gen);
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }
}

/// Save a genome to a JSON file.
pub fn save_genome(path: impl AsRef<Path>, saved: &SavedGenome) -> io::Result<()> {
    let json = serde_json::to_string_pretty(saved)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, json)
}

/// Load a genome from a JSON file.
pub fn load_genome(path: impl AsRef<Path>) -> io::Result<SavedGenome> {
    let json = fs::read_to_string(path)?;
    serde_json::from_str(&json).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Save evolution statistics to a JSON file.
pub fn save_stats(path: impl AsRef<Path>, stats: &[GenStats]) -> io::Result<()> {
    // GenStats doesn't derive Serialize, so we convert manually
    let records: Vec<serde_json::Value> = stats
        .iter()
        .map(|s| {
            serde_json::json!({
                "generation": s.generation,
                "best_fitness": s.best_fitness,
                "avg_fitness": s.avg_fitness,
                "avg_size": s.avg_size,
                "best_program": s.best_program,
            })
        })
        .collect();
    let json = serde_json::to_string_pretty(&records)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    fs::write(path, json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, Node};

    #[test]
    fn test_save_load_genome() {
        let tree = Node::BinOp(
            BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::IntConst(1)),
        );
        let saved = SavedGenome::new(tree.clone())
            .with_fitness(0.001)
            .with_generation(42)
            .with_description("test genome");

        let path = std::env::temp_dir().join("genlang_test_genome.json");
        save_genome(&path, &saved).unwrap();

        let loaded = load_genome(&path).unwrap();
        assert_eq!(loaded.genome, tree);
        assert_eq!(loaded.fitness, Some(0.001));
        assert_eq!(loaded.generation, Some(42));
        assert_eq!(loaded.description.as_deref(), Some("test genome"));

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_stats() {
        let stats = vec![GenStats {
            generation: 0,
            best_fitness: 1.0,
            avg_fitness: 5.0,
            avg_size: 10.0,
            best_program: "x0".into(),
        }];
        let path = std::env::temp_dir().join("genlang_test_stats.json");
        save_stats(&path, &stats).unwrap();

        let content = fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"generation\": 0"));
        fs::remove_file(&path).ok();
    }
}
