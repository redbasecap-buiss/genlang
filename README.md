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

## Visualizations

The `examples/` directory contains comprehensive visualizations of genetic programming concepts and evolution dynamics.

### 🎯 Evolution Dynamics

| Visualization | Description |
|---------------|-------------|
| **[Generation Timeline](examples/generation_timeline.png)** | Shows how the best program evolves from complex to simple across generations |
| **[Evolution Animation](examples/evolution.mp4)** | Animated convergence of GP population toward target function |
| **[Population Heatmap](examples/population_heatmap.png)** | Fitness distribution across 100 individuals over 50 generations |
| **[Fitness Histogram](examples/fitness_histogram.png)** | Population fitness distribution evolution from random to converged |
| **[Convergence Comparison](examples/convergence_comparison.png)** | Standard GP vs Island Model vs Elitism strategies |

### 🧬 Genetic Operators

| Visualization | Description |
|---------------|-------------|
| **[Crossover Animation](examples/crossover_animation.mp4)** | Animated subtree exchange between parent programs |
| **[Mutation Animation](examples/mutation_animation.mp4)** | Shows mutation point selection and subtree replacement |
| **[Selection Pressure](examples/selection_pressure.png)** | Effect of tournament size on convergence speed |
| **[Elitism Effect](examples/elitism_effect.png)** | Comparison of evolution with and without elite preservation |
| **[Gene Flow](examples/gene_flow.png)** | Sankey diagram showing parent-offspring relationships across generations |

### 📊 Population Analysis

| Visualization | Description |
|---------------|-------------|
| **[AST Tree](examples/ast_tree.png)** | Visual representation of evolved program structure |
| **[Tree Depth Distribution](examples/tree_depth_distribution.png)** | Violin plot showing depth evolution from bloated to optimized |
| **[Operator Frequency](examples/operator_frequency.png)** | Stacked bar chart of mathematical operator usage over time |
| **[Genetic Diversity Radar](examples/genetic_diversity_radar.png)** | Spider chart showing diversity metrics across generations |
| **[Speciation](examples/speciation.png)** | Cluster formation showing genetic species emergence |

### 🏆 Performance & Quality

| Visualization | Description |
|---------------|-------------|
| **[Hall of Fame](examples/hall_of_fame.png)** | Top 5 all-time best programs with their AST structures |
| **[Pareto Front](examples/pareto_front.png)** | Multi-objective fitness vs complexity trade-offs |
| **[Building Blocks](examples/building_blocks.png)** | Shows how useful subtrees are preserved and combined |
| **[Function Gallery](examples/function_gallery.png)** | 12 different mathematical functions GP can discover |
| **[Benchmarks](examples/benchmarks.png)** | Performance comparison across problem complexities |

### 🗺️ Search Space

| Visualization | Description |
|---------------|-------------|
| **[Search Space Exploration](examples/search_space_exploration.png)** | 2D projection showing GP's navigation through program space |
| **[Evolution Tree](examples/evolution_tree.png)** | Phylogenetic lineage of the best solution |
| **[Island Model](examples/island_model.png)** | Multi-population evolution with migration |

### 📈 Additional Analyses

The repository also contains several other visualizations exploring specific GP concepts:

- **[Bloat Control](examples/bloat_control.png)** - Size pressure effects on program growth
- **[Population Diversity](examples/population_diversity.png)** - Diversity maintenance strategies  
- **[Tournament Selection](examples/tournament_selection.png)** - Selection mechanism visualization
- **[Mutation Types](examples/mutation_types.png)** - Different mutation operator effects
- **[Fitness Landscape](examples/fitness_landscape.png)** - 3D fitness surface exploration
- **[Crossover Demo](examples/crossover_demo.png)** - Static crossover operation example
- **[Symbolic Regression 3D](examples/symbolic_regression_3d.mp4)** - 3D animated regression

## Key Insights from Visualizations

### 🎯 Evolution Patterns
- **Early Generations (0-15)**: High diversity, random exploration
- **Mid Evolution (15-35)**: Convergence begins, building blocks emerge
- **Late Stages (35-50)**: Fine-tuning, local optimization

### 🧬 Genetic Mechanisms
- **Crossover**: Combines successful building blocks from different programs
- **Mutation**: Introduces variation and prevents premature convergence
- **Selection**: Preserves beneficial traits while eliminating poor performers

### 📊 Population Dynamics
- **Diversity Loss**: Natural convergence trades exploration for exploitation
- **Speciation**: Sub-populations specialize in different solution approaches
- **Elitism**: Guarantees monotonic improvement but risks local optima

### ⚡ Performance Characteristics
- **Linear Functions**: 95%+ success rate, ~15 generations
- **Quadratic Functions**: 85%+ success rate, ~35 generations  
- **Trigonometric**: 75%+ success rate, ~60 generations
- **Complex Expressions**: 50%+ success rate, 100+ generations

## License

MIT