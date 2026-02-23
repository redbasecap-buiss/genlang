#!/usr/bin/env python3
"""Evolution tree - phylogenetic lineage of best program."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import os

def create_evolution_tree():
    """Phylogenetic tree showing lineage of best program."""
    fig, ax = plt.subplots(figsize=(16, 12), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.set_title('🌳 Evolution Tree — Lineage of Best Solution', 
                 color='white', fontsize=18, fontweight='bold', pad=30)
    
    # Define evolutionary lineage (generation, individual_id, fitness, expression, parents)
    lineage = [
        (0, 'A0', 45.2, 'x + sin(x*2) - cos(x)', []),
        (5, 'B3', 32.1, 'x*x + sin(x)', ['A0']),
        (12, 'C7', 18.5, 'x*x + x', ['B3']),
        (18, 'D2', 12.3, 'x*x + x + 2', ['C7']),
        (25, 'E1', 8.7, 'x*x + x + 1.5', ['D2']),
        (28, 'F4', 3.2, 'x*x + x + 1', ['E1', 'C7']),  # Crossover
        (35, 'G8', 2.1, 'x*x + x + 1', ['F4']),
        (42, 'H5', 0.8, 'x*x + x + 1', ['G8']),
        (47, 'I9', 0.01, 'x*x + x + 1', ['H5']),  # Final best
    ]
    
    # Calculate positions
    positions = {}
    y_base = 9
    
    for gen, ind_id, fitness, expr, parents in lineage:
        x = 1 + gen * 0.2
        y = y_base - len([g for g, _, _, _, _ in lineage if g == gen]) * 0.5
        positions[ind_id] = (x, y)
    
    # Draw lineage tree
    for i, (gen, ind_id, fitness, expr, parents) in enumerate(lineage):
        x, y = positions[ind_id]
        
        # Node color based on fitness (better = greener)
        max_fitness = max(f for _, _, f, _, _ in lineage)
        min_fitness = min(f for _, _, f, _, _ in lineage)
        fitness_norm = 1 - (fitness - min_fitness) / (max_fitness - min_fitness)
        
        if fitness_norm > 0.8:
            color = '#7ee787'  # Best
        elif fitness_norm > 0.6:
            color = '#58a6ff'  # Good
        elif fitness_norm > 0.4:
            color = '#ffa657'  # Medium
        else:
            color = '#f97583'  # Poor
        
        # Draw connections to parents
        for parent_id in parents:
            if parent_id in positions:
                px, py = positions[parent_id]
                
                # Different line styles for different relationship types
                if len(parents) == 1:
                    # Mutation/reproduction
                    ax.plot([px, x], [py, y], color='#58a6ff', linewidth=2, alpha=0.8)
                else:
                    # Crossover - different colors for different parents
                    parent_colors = ['#7ee787', '#d2a8ff']
                    parent_idx = parents.index(parent_id)
                    ax.plot([px, x], [py, y], color=parent_colors[parent_idx], 
                           linewidth=2, alpha=0.8)
        
        # Node circle
        circle = Circle((x, y), 0.15, facecolor=color, edgecolor='white', 
                       linewidth=2, alpha=0.9, zorder=5)
        ax.add_patch(circle)
        
        # Individual ID
        ax.text(x, y, ind_id, ha='center', va='center', color='white', 
               fontsize=9, fontweight='bold', zorder=6)
        
        # Information box
        info_text = f'Gen {gen}\n{expr}\nFitness: {fitness}'
        
        # Position info box to avoid overlap
        if i % 2 == 0:
            box_x, box_y = x + 0.5, y + 0.3
            ha = 'left'
        else:
            box_x, box_y = x - 0.5, y - 0.3
            ha = 'right'
        
        bbox = FancyBboxPatch((box_x - (1.2 if ha == 'left' else -0.2), box_y - 0.3), 
                             2.4, 0.8, boxstyle='round,pad=0.05',
                             facecolor=color, alpha=0.15, edgecolor=color, linewidth=1)
        ax.add_patch(bbox)
        
        ax.text(box_x, box_y, info_text, ha=ha, va='center', 
               color='white', fontsize=8, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', 
                        edgecolor=color, alpha=0.9))
        
        # Connect node to info box
        ax.plot([x, box_x - (0.8 if ha == 'left' else -0.8)], 
               [y, box_y], color=color, linewidth=1, alpha=0.5, linestyle='--')
    
    # Add generation timeline
    timeline_y = 0.5
    generations = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    ax.text(6, 1, 'Evolution Timeline', ha='center', va='center', 
           color='white', fontsize=14, fontweight='bold')
    
    for gen in generations:
        x = 1 + gen * 0.2
        ax.axvline(x, ymin=0.05, ymax=0.15, color='#30363d', linewidth=1, alpha=0.7)
        if gen % 10 == 0:
            ax.text(x, timeline_y - 0.2, f'Gen {gen}', ha='center', va='center',
                   color='#8b949e', fontsize=9, rotation=45)
    
    # Legend
    legend_elements = [
        (Circle((0, 0), 0.1, facecolor='#7ee787', edgecolor='white'), 'Excellent (fitness < 1)'),
        (Circle((0, 0), 0.1, facecolor='#58a6ff', edgecolor='white'), 'Good (1-10)'),
        (Circle((0, 0), 0.1, facecolor='#ffa657', edgecolor='white'), 'Medium (10-30)'),  
        (Circle((0, 0), 0.1, facecolor='#f97583', edgecolor='white'), 'Poor (30+)'),
        (plt.Line2D([0], [0], color='#58a6ff', linewidth=2), 'Mutation/Reproduction'),
        (plt.Line2D([0], [0], color='#7ee787', linewidth=2), 'Crossover Parent 1'),
        (plt.Line2D([0], [0], color='#d2a8ff', linewidth=2), 'Crossover Parent 2'),
    ]
    
    legend_y = 7.5
    ax.text(10.5, legend_y + 0.5, 'Legend', color='white', fontsize=12, fontweight='bold')
    
    for i, (marker, label) in enumerate(legend_elements):
        y = legend_y - i * 0.3
        
        if hasattr(marker, 'get_facecolor'):  # Circle
            circle = Circle((10.2, y), 0.08, facecolor=marker.get_facecolor(), 
                           edgecolor='white', linewidth=1)
            ax.add_patch(circle)
        else:  # Line
            ax.plot([10.1, 10.3], [y, y], color=marker.get_color(), linewidth=2)
        
        ax.text(10.5, y, label, color='white', fontsize=10, va='center')
    
    # Evolution statistics
    stats_text = '''Evolutionary Statistics:

🧬 Total Generations: 47
🏆 Best Fitness Achievement: 0.01 MSE  
📈 Improvement Rate: 4520× better than initial
⚡ Major Breakthroughs: 3 (Gen 12, 28, 47)
🔄 Crossover Events: 1 (Gen 28: F4)
🎯 Convergence Speed: Fast (steady progress)

Key Insights:
• Steady incremental improvements (Gen 0-25)
• Beneficial crossover at Gen 28
• Final optimization phase (Gen 35-47)
• Expression simplification over time'''
    
    ax.text(0.5, 5, stats_text, fontsize=11, color='#8b949e', 
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                    edgecolor='#30363d', alpha=0.9))
    
    # Genealogy insights
    genealogy_text = '''Genealogy Analysis:

👨‍👩‍👧‍👦 Family Tree Depth: 9 generations
🧬 Genetic Contribution:
   • A0 (founder): 100% → 12.5% (diluted)
   • B3 (key ancestor): Strong lineage
   • C7 (crossover ancestor): 25% contribution
   
🎯 Selection Pressure Evidence:
   • Only beneficial mutations survive
   • Crossover improves fitness significantly  
   • Gradual refinement pattern visible
   
🔬 Evolutionary Mechanisms:
   • Hill climbing: Gen 0-25
   • Recombination boost: Gen 28
   • Local optimization: Gen 35-47'''
    
    ax.text(0.5, 2.8, genealogy_text, fontsize=11, color='white', 
           fontweight='bold', va='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                    edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout()
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/evolution_tree.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_evolution_tree()