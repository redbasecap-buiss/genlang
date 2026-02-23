#!/usr/bin/env python3
"""Fitness histogram evolution across generations."""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_fitness_histogram():
    """Multi-panel histogram of population fitness evolution."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0d1117')
    fig.suptitle('📊 Population Fitness Distribution Evolution', 
                 color='white', fontsize=18, fontweight='bold', y=0.95)
    
    # Flatten axes for easier access
    axes = axes.flatten()
    
    # Generation snapshots
    generations = [0, 10, 25, 50]
    colors = ['#f97583', '#ffa657', '#7ee787', '#58a6ff']
    
    np.random.seed(42)
    
    # Generate fitness data for each generation
    fitness_data = []
    
    for i, gen in enumerate(generations):
        if gen == 0:
            # Initial generation: wide, random distribution
            fitness = np.random.exponential(scale=8, size=200) + 1
            fitness = np.concatenate([
                np.random.exponential(scale=5, size=150) + 2,
                np.random.exponential(scale=15, size=50) + 10
            ])
        else:
            # Evolution progress
            progress = i / (len(generations) - 1)
            
            # Selection bias toward better fitness
            n_individuals = 200
            
            # Elite group (getting larger and better)
            n_elite = int(20 + progress * 50)
            elite_fitness = np.random.gamma(2, 0.5 + progress * 0.3, n_elite) + 0.1
            
            # Regular population (getting more focused)
            n_regular = n_individuals - n_elite
            regular_mean = 8 - progress * 6
            regular_std = 4 - progress * 2.5
            regular_fitness = np.random.normal(regular_mean, regular_std, n_regular)
            regular_fitness = np.maximum(regular_fitness, 0.5)
            
            fitness = np.concatenate([elite_fitness, regular_fitness])
            
        fitness_data.append(fitness)
    
    # Create histograms
    for i, (ax, gen, fitness, color) in enumerate(zip(axes, generations, fitness_data, colors)):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        
        # Histogram
        n, bins, patches = ax.hist(fitness, bins=25, color=color, alpha=0.7, 
                                  edgecolor='white', linewidth=0.5, density=True)
        
        # Color gradient for bars (lower fitness = greener)
        for j, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
            bin_center = (bin_left + bin_right) / 2
            # Color based on fitness (lower = better = greener)
            if bin_center < np.percentile(fitness, 20):  # Best 20%
                patch.set_facecolor('#7ee787')
            elif bin_center < np.percentile(fitness, 50):  # Good 50%
                patch.set_facecolor('#58a6ff')
            elif bin_center < np.percentile(fitness, 80):  # Average 80%
                patch.set_facecolor('#ffa657')
            else:  # Worst 20%
                patch.set_facecolor('#f97583')
            patch.set_alpha(0.8)
        
        # Statistics
        mean_fit = np.mean(fitness)
        median_fit = np.median(fitness)
        std_fit = np.std(fitness)
        best_fit = np.min(fitness)
        
        # Vertical lines for statistics
        ax.axvline(mean_fit, color='white', linestyle='-', linewidth=2, alpha=0.8, label='Mean')
        ax.axvline(median_fit, color='#d2a8ff', linestyle='--', linewidth=2, alpha=0.8, label='Median')
        ax.axvline(best_fit, color='#7ee787', linestyle=':', linewidth=3, alpha=0.9, label='Best')
        
        # Title and labels
        ax.set_title(f'Generation {gen}', color='white', fontsize=14, fontweight='bold')
        ax.set_xlabel('Fitness (MSE)', color='white', fontsize=12)
        ax.set_ylabel('Density', color='white', fontsize=12)
        ax.grid(True, alpha=0.3, color='#30363d')
        
        # Statistics text box
        stats_text = f'Mean: {mean_fit:.2f}\nStd: {std_fit:.2f}\nBest: {best_fit:.3f}\nN: {len(fitness)}'
        ax.text(0.75, 0.75, stats_text, transform=ax.transAxes, 
                fontsize=10, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', 
                         edgecolor='#30363d', alpha=0.9))
        
        # Show fitness improvement over time
        if i > 0:
            prev_best = np.min(fitness_data[i-1])
            improvement = ((prev_best - best_fit) / prev_best) * 100
            ax.text(0.05, 0.85, f'Improvement:\n{improvement:.1f}%', 
                   transform=ax.transAxes, fontsize=10, color='#7ee787', 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', 
                            edgecolor='#7ee787', alpha=0.8))
        
        # Legend for first plot
        if i == 0:
            ax.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d', 
                     labelcolor='white', fontsize=10)
    
    # Add evolution arrows between plots
    # Arrow from gen 0 to gen 10
    fig.text(0.48, 0.75, '→', fontsize=30, color='white', ha='center', va='center')
    # Arrow from gen 10 to gen 25  
    fig.text(0.52, 0.42, '→', fontsize=30, color='white', ha='center', va='center')
    # Arrow from gen 25 to gen 50
    fig.text(0.48, 0.42, '→', fontsize=30, color='white', ha='center', va='center')
    
    # Overall evolution summary
    evolution_summary = '''Evolution Pattern:

Gen 0:  Wide, random distribution
        High diversity, poor average fitness
        
Gen 10: Slight improvement visible
        Some individuals finding better solutions
        
Gen 25: Clear convergence beginning
        Elite group emerging
        Distribution narrowing
        
Gen 50: Strong convergence
        Most individuals near optimum
        Clear fitness improvement'''
    
    fig.text(0.02, 0.02, evolution_summary, fontsize=12, color='#8b949e', 
             fontweight='bold', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    # Key metrics comparison
    initial_mean = np.mean(fitness_data[0])
    final_mean = np.mean(fitness_data[-1])
    initial_best = np.min(fitness_data[0])
    final_best = np.min(fitness_data[-1])
    
    metrics_comparison = f'''Performance Metrics:

                 Gen 0    →    Gen 50    Change
              ────────────────────────────────────
Population Mean:  {initial_mean:.2f}    →    {final_mean:.2f}     {((final_mean-initial_mean)/initial_mean)*100:+.0f}%
Best Individual:  {initial_best:.2f}    →    {final_best:.3f}    {((final_best-initial_best)/initial_best)*100:+.0f}%
Std Deviation:    {np.std(fitness_data[0]):.2f}    →    {np.std(fitness_data[-1]):.2f}     {((np.std(fitness_data[-1])-np.std(fitness_data[0]))/np.std(fitness_data[0]))*100:+.0f}%

Overall Improvement: {((initial_best - final_best) / initial_best)*100:.0f}% better solution found'''
    
    fig.text(0.55, 0.02, metrics_comparison, fontsize=11, color='white', 
             fontweight='bold', va='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.92])
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/fitness_histogram.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_fitness_histogram()