#!/usr/bin/env python3
"""Gene flow Sankey diagram."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import os

def create_gene_flow():
    """Sankey-style diagram showing gene flow across generations."""
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    ax.set_title('🧬 Gene Flow — Parent to Offspring Connections', 
                 color='white', fontsize=18, fontweight='bold', pad=30)
    
    # Define generations
    gen_positions = [1, 3, 5, 7, 9]
    gen_labels = ['Gen 0', 'Gen 1', 'Gen 2', 'Gen 3', 'Gen 4']
    
    # Population sizes (decreasing elite population)
    pop_sizes = [10, 8, 6, 4, 2]
    colors = ['#f97583', '#ffa657', '#7ee787', '#58a6ff', '#d2a8ff']
    
    # Draw generation columns
    individuals = {}  # Store individual positions for flow lines
    
    for i, (x_pos, label, pop_size, color) in enumerate(zip(gen_positions, gen_labels, pop_sizes, colors)):
        # Generation label
        ax.text(x_pos, 7.5, label, ha='center', va='center', color='white', 
                fontsize=14, fontweight='bold')
        
        # Individual boxes
        y_start = 4 - (pop_size - 1) * 0.3
        individuals[i] = []
        
        for j in range(pop_size):
            y_pos = y_start + j * 0.6
            individuals[i].append((x_pos, y_pos))
            
            # Fitness-based color intensity
            fitness_score = np.random.rand() * 0.8 + 0.2  # 0.2 to 1.0
            alpha = 0.3 + 0.7 * fitness_score
            
            rect = FancyBboxPatch((x_pos - 0.15, y_pos - 0.15), 0.3, 0.3,
                                 boxstyle='round,pad=0.02', 
                                 facecolor=color, edgecolor='white', 
                                 linewidth=1, alpha=alpha)
            ax.add_patch(rect)
            
            # Individual ID
            ax.text(x_pos, y_pos, f'{j+1}', ha='center', va='center', 
                   color='white', fontsize=10, fontweight='bold')
    
    # Create flow lines (parent-child relationships)
    np.random.seed(42)
    
    for gen in range(len(gen_positions) - 1):
        current_gen = individuals[gen]
        next_gen = individuals[gen + 1]
        
        # Each child has 1-2 parents
        for child_idx, (child_x, child_y) in enumerate(next_gen):
            n_parents = np.random.choice([1, 2], p=[0.3, 0.7])  # Mostly crossover
            
            # Select parents (fitness-biased)
            if len(current_gen) <= 4:
                parent_probs = np.array([0.4, 0.3, 0.2, 0.1])[:len(current_gen)]
            else:
                parent_probs = np.array([0.4, 0.3, 0.2, 0.1])
                parent_probs = np.concatenate([parent_probs, np.full(len(current_gen) - 4, 0.1 / (len(current_gen) - 4))])
            parent_probs = parent_probs / parent_probs.sum()
            
            parent_indices = np.random.choice(len(current_gen), size=n_parents, 
                                            replace=False, p=parent_probs)
            
            for parent_idx in parent_indices:
                parent_x, parent_y = current_gen[parent_idx]
                
                # Flow line properties
                if n_parents == 1:
                    # Mutation/reproduction
                    line_color = '#58a6ff'
                    alpha = 0.6
                    linewidth = 2
                else:
                    # Crossover
                    line_color = '#7ee787'
                    alpha = 0.4
                    linewidth = 1.5
                
                # Curved flow line
                mid_x = (parent_x + child_x) / 2
                control_points = [
                    [parent_x + 0.15, parent_y],
                    [mid_x, (parent_y + child_y) / 2 + np.random.uniform(-0.2, 0.2)],
                    [child_x - 0.15, child_y]
                ]
                
                # Draw smooth curve using multiple line segments
                n_segments = 20
                curve_x = []
                curve_y = []
                
                for t in np.linspace(0, 1, n_segments):
                    # Bezier curve
                    x = (1-t)**2 * control_points[0][0] + 2*(1-t)*t * control_points[1][0] + t**2 * control_points[2][0]
                    y = (1-t)**2 * control_points[0][1] + 2*(1-t)*t * control_points[1][1] + t**2 * control_points[2][1]
                    curve_x.append(x)
                    curve_y.append(y)
                
                ax.plot(curve_x, curve_y, color=line_color, linewidth=linewidth, 
                       alpha=alpha, zorder=1)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#58a6ff', alpha=0.6, label='Mutation/Copy'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#7ee787', alpha=0.4, label='Crossover'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#f97583', alpha=0.7, label='High Fitness'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#f97583', alpha=0.3, label='Low Fitness'),
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(0.02, 0.98), facecolor='#21262d', 
                      edgecolor='#30363d', labelcolor='white', fontsize=11)
    legend.get_frame().set_alpha(0.9)
    
    # Add statistics boxes
    stats_boxes = [
        (1, 0.5, 'Initial Pop\n100 random\nprograms', '#f97583'),
        (3, 0.5, 'Selection\n60% survival\nrate', '#ffa657'),
        (5, 0.5, 'Crossover\n70% of new\nindividuals', '#7ee787'),
        (7, 0.5, 'Elite\nTop 20%\npreserved', '#58a6ff'),
        (9, 0.5, 'Convergence\nBest solution\nfound', '#d2a8ff'),
    ]
    
    for x, y, text, color in stats_boxes:
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.2, 
                   edgecolor=color, linewidth=2)
        ax.text(x, y, text, ha='center', va='center', color='white', 
               fontsize=9, fontweight='bold', bbox=bbox)
    
    # Flow explanation
    ax.text(5, 0.1, 'Gene Flow Direction ➤', ha='center', va='bottom', 
           color='#8b949e', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/gene_flow.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_gene_flow()