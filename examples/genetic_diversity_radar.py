#!/usr/bin/env python3
"""Genetic diversity radar chart."""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_genetic_diversity_radar():
    """Radar chart showing diversity metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0d1117', 
                            subplot_kw=dict(projection='polar'))
    fig.suptitle('🕸️ Genetic Diversity Metrics — Population Health Over Time', 
                 color='white', fontsize=16, fontweight='bold', y=0.95)
    
    # Define diversity metrics
    metrics = ['Tree Depth\nVariance', 'Operator\nDiversity', 'Constant\nRange', 
               'Fitness\nSpread', 'Structural\nComplexity', 'Function\nTypes']
    n_metrics = len(metrics)
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Early generation data (high diversity)
    early_values = [0.9, 0.85, 0.92, 0.88, 0.91, 0.87]  # High diversity across metrics
    early_values += early_values[:1]
    
    # Late generation data (lower diversity due to convergence)
    late_values = [0.4, 0.3, 0.5, 0.2, 0.35, 0.25]  # Lower diversity
    late_values += late_values[:1]
    
    # Optimal diversity (balanced)
    optimal_values = [0.6, 0.7, 0.65, 0.4, 0.55, 0.6]  # Moderate diversity
    optimal_values += optimal_values[:1]
    
    for ax, title, values, color, alpha in [
        (axes[0], 'Early Generation (Diverse)', early_values, '#f97583', 0.3),
        (axes[1], 'Late Generation (Converged)', late_values, '#58a6ff', 0.3)
    ]:
        ax.set_facecolor('#0d1117')
        ax.grid(True, color='#30363d', alpha=0.6)
        ax.set_ylim(0, 1)
        
        # Plot the diversity polygon
        ax.plot(angles, values, 'o-', linewidth=2, color=color, alpha=0.8)
        ax.fill(angles, values, color=color, alpha=alpha)
        
        # Plot optimal diversity for comparison
        ax.plot(angles, optimal_values, '--', linewidth=2, color='#7ee787', alpha=0.7)
        ax.fill(angles, optimal_values, color='#7ee787', alpha=0.1)
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, color='white', fontsize=10, fontweight='bold')
        
        # Customize radial labels
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                          color='#8b949e', fontsize=9)
        
        # Title
        ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=20)
        
        # Add value labels on the plot
        for angle, value, metric in zip(angles[:-1], values[:-1], metrics):
            x = angle
            y = value + 0.05
            ax.text(x, y, f'{value:.2f}', ha='center', va='center', 
                   color='white', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='#f97583', linewidth=2, alpha=0.8, label='Early Gen'),
        plt.Line2D([0], [0], color='#58a6ff', linewidth=2, alpha=0.8, label='Late Gen'),
        plt.Line2D([0], [0], color='#7ee787', linewidth=2, alpha=0.7, 
                   linestyle='--', label='Optimal Zone'),
    ]
    
    axes[1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.1, 1.1),
                  facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=11)
    
    # Add detailed analysis
    analysis_text = '''Diversity Analysis:

Early Generation:
• High exploration phase
• Wide range of solutions
• Risk of bloat and inefficiency

Late Generation:  
• Convergence to optima
• Reduced exploration
• Risk of premature convergence

Optimal Balance:
• Moderate diversity maintained
• Exploitation with exploration
• Sustainable evolution'''
    
    fig.text(0.02, 0.02, analysis_text, fontsize=11, color='#8b949e', 
             fontweight='bold', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    # Evolution timeline
    timeline_text = '''Evolution Timeline:

Gen 0-10:   High diversity exploration
Gen 10-25:  Gradual specialization  
Gen 25-40:  Selection pressure increases
Gen 40+:    Convergence and refinement

Diversity Management:
✓ Monitor fitness spread
✓ Control bloat growth
✓ Maintain operator variety
✓ Preserve building blocks'''
    
    fig.text(0.65, 0.02, timeline_text, fontsize=11, color='white', 
             fontweight='bold', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    # Add diversity metrics explanation
    metrics_explanation = '''Diversity Metrics Explained:

🌳 Tree Depth Variance: Structural variety in program sizes
📊 Operator Diversity: Range of mathematical functions used  
🔢 Constant Range: Spread of numerical values
📈 Fitness Spread: Performance variation in population
🏗️  Structural Complexity: AST shape and organization diversity
⚙️  Function Types: Variety of mathematical operations

Higher values = more diverse population
Lower values = more converged/specialized'''
    
    fig.text(0.35, 0.02, metrics_explanation, fontsize=10, color='#8b949e', 
             fontweight='bold', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.92])
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/genetic_diversity_radar.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_genetic_diversity_radar()