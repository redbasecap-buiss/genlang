#!/usr/bin/env python3
"""Tree depth distribution across generations."""

import numpy as np
import matplotlib.pyplot as plt
# from scipy import stats  # Not needed for this visualization
import os

def create_tree_depth_distribution():
    """Violin plot of tree depths across generations."""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_color('#30363d')
    
    np.random.seed(42)
    
    # Generate tree depth data across generations
    generations = [0, 10, 20, 30, 40, 50]
    gen_colors = ['#f97583', '#ffa657', '#7ee787', '#58a6ff', '#d2a8ff', '#ff6b9d']
    
    depth_data = []
    
    for i, gen in enumerate(generations):
        # Initial generation: random depths (wide distribution)
        if gen == 0:
            depths = np.random.gamma(3, 2) + 2  # Starts around 4-10 depth
            probs = np.exp(-np.arange(2, 15) * 0.3)
            probs = probs / probs.sum()
            depths = np.random.choice(range(2, 15), 100, p=probs)
            depths = depths + np.random.normal(0, 0.5, 100)
        else:
            # Evolution tends toward optimal depth (around 6-8)
            progress = i / (len(generations) - 1)
            
            # Target depth becomes more focused
            target_depth = 7 - progress * 1  # Slight decrease over time
            spread = 3 - progress * 2  # Narrowing distribution
            
            # Generate depths with selection pressure
            depths = np.random.normal(target_depth, spread, 100)
            depths = np.clip(depths, 1, 20)  # Reasonable bounds
            
            # Add some bloat (occasional deep trees)
            if np.random.rand() < 0.3:
                n_bloated = np.random.randint(5, min(15, len(depths)))
                bloated_depths = np.random.exponential(5, n_bloated) + 10
                depths[:n_bloated] = bloated_depths
        
        depth_data.append(depths)
    
    # Create violin plots
    violin_parts = ax.violinplot(depth_data, positions=generations, widths=6,
                                showmeans=True, showmedians=True, showextrema=True)
    
    # Color each violin
    for i, (pc, color) in enumerate(zip(violin_parts['bodies'], gen_colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor('white')
        pc.set_linewidth(1)
    
    # Style the statistical lines
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in violin_parts:
            vp = violin_parts[partname]
            vp.set_edgecolor('white')
            vp.set_linewidth(2)
    
    # Add individual data points (jittered)
    for i, (gen, depths, color) in enumerate(zip(generations, depth_data, gen_colors)):
        # Jitter for visibility
        jittered_x = gen + np.random.normal(0, 1.5, len(depths))
        ax.scatter(jittered_x, depths, alpha=0.3, s=15, color=color, 
                  edgecolors='white', linewidth=0.5, zorder=3)
    
    # Calculate and show statistics
    for i, (gen, depths, color) in enumerate(zip(generations, depth_data, gen_colors)):
        mean_depth = np.mean(depths)
        median_depth = np.median(depths)
        std_depth = np.std(depths)
        
        # Stats box
        stats_text = f'Gen {gen}\nμ={mean_depth:.1f}\nσ={std_depth:.1f}'
        ax.text(gen, max(depths) + 1, stats_text, ha='center', va='bottom', 
               color='white', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3, 
                        edgecolor=color))
    
    # Trend line (mean depth over generations)
    mean_depths = [np.mean(depths) for depths in depth_data]
    ax.plot(generations, mean_depths, color='white', linewidth=3, 
           marker='o', markersize=8, alpha=0.8, zorder=5, label='Mean Depth Trend')
    
    # Styling
    ax.set_xlabel('Generation', color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Tree Depth', color='white', fontsize=14, fontweight='bold')
    ax.set_title('🌳 Tree Depth Distribution Evolution — From Deep to Optimized', 
                 color='white', fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, color='#30363d')
    ax.set_ylim(0, 25)
    
    # Add optimal depth zone
    ax.axhspan(5, 9, alpha=0.1, color='#7ee787', zorder=1)
    ax.text(25, 7, 'Optimal Depth Zone', ha='center', va='center', 
           color='#7ee787', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', 
                    edgecolor='#7ee787', alpha=0.8))
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='white', linewidth=3, marker='o', 
                   label='Mean Depth Trend'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#7ee787', alpha=0.1, 
                     label='Optimal Zone'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='#21262d', 
              edgecolor='#30363d', labelcolor='white', fontsize=11)
    
    # Evolution phases annotations
    phases = [
        (5, 22, 'Random\nInitialization', '#f97583'),
        (15, 22, 'Selection\nPressure', '#ffa657'),
        (25, 22, 'Bloat\nControl', '#7ee787'),
        (35, 22, 'Convergence', '#58a6ff'),
        (45, 22, 'Optimization', '#d2a8ff'),
    ]
    
    for x, y, label, color in phases:
        ax.annotate(label, xy=(x, 15), xytext=(x, y),
                   color=color, fontsize=10, fontweight='bold', ha='center',
                   arrowprops=dict(arrowstyle='->', color=color, alpha=0.7))
    
    # Statistics summary
    initial_mean = np.mean(depth_data[0])
    final_mean = np.mean(depth_data[-1])
    reduction = ((initial_mean - final_mean) / initial_mean) * 100
    
    summary = (f'Depth Evolution Summary:\n'
              f'• Initial Mean: {initial_mean:.1f}\n'
              f'• Final Mean: {final_mean:.1f}\n'
              f'• Reduction: {reduction:.1f}%\n'
              f'• Convergence: {abs(np.std(mean_depths)):.2f}')
    
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                     edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout()
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/tree_depth_distribution.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_tree_depth_distribution()