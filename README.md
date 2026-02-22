# 🧬 genlang

**Self-evolving programming language** — programs mutate, crossover, and evolve to solve problems. Genetic Programming in pure Rust.

## Features

- **AST-based genome**: arithmetic, comparison, conditionals, variables, constants, math functions
- **Random program generation** with configurable depth
- **GP operators**: subtree crossover, point/hoist/shrink mutation
- **Tournament selection** with elitism and bloat control
- **Sandboxed interpreter**: step-limited (10k), protected division, safe math
- **Symbolic regression**: evolve equations from data
- **Serialization**: save/load genomes as JSON

## Quick Start

```bash
cargo run    # Demo: evolve x² + x + 1
cargo test   # Run all tests
```

## Architecture

```
src/
├── ast.rs          # AST node types, random generation, serialization
├── interpreter.rs  # Stack-based VM with step limits
├── genetic.rs      # GP engine: selection, crossover, mutation, evolution
├── problems.rs     # Problem definitions (symbolic regression, etc.)
├── lib.rs          # Library root
└── main.rs         # Demo entry point
```

## License

MIT
