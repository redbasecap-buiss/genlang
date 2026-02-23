#!/usr/bin/env python3
"""Speciation - cluster visualization of GP population."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# from sklearn.cluster import KMeans  # Not needed for this visualization
import os

def create_speciation():
    """Cluster visualization showing species forming in GP population."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0d1117')
    fig.suptitle('🧬 Speciation — Genetic Clustering in GP Population', 
                 color='white', fontsize=16, fontweight='bold', y=0.95)
    
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
    
    axes[0].set_title('Early Generation (Diverse)', color='white', fontsize=12, fontweight='bold')
    axes[1].set_title('Later Generation (Specialized)', color='white', fontsize=12, fontweight='bold')
    
    np.random.seed(42)
    
    # Generate synthetic GP population data
    # Dimensions represent: tree_depth, n_operators, avg_constant_value, fitness
    
    # Early generation - more diverse, scattered
    n_individuals = 200
    
    # Create multiple random clusters (diverse population)
    early_data = []
    for _ in range(n_individuals):
        # Random diversity
        centers = [[-3, 2], [2, 3], [-2, -2], [3, -1], [0, 0]]
        center = centers[np.random.choice(len(centers))]
        point = np.random.randn(2) * 2 + np.array(center) + np.random.randn(2) * 0.8
        early_data.append(point)
    early_data = np.array(early_data)
    
    # Later generation - more clustered (species formation)
    late_data = []
    species_centers = [
        [-2, 2, '#7ee787'],    # Polynomial species
        [2, -1, '#58a6ff'],    # Trigonometric species  
        [0, 1.5, '#f97583'],   # Mixed species
        [1, -2, '#d2a8ff'],    # Linear species
        [-1, -1, '#ffa657'],   # Exponential species
    ]
    
    for center_x, center_y, color in species_centers:
        # Generate cluster around center
        cluster_size = np.random.randint(25, 50)
        cluster_data = np.random.multivariate_normal(
            [center_x, center_y], 
            [[0.3, 0.1], [0.1, 0.3]], 
            cluster_size
        )
        
        for point in cluster_data:
            late_data.append([point[0], point[1], color])
    
    # Plot early generation (scattered)
    ax = axes[0]
    ax.scatter(early_data[:, 0], early_data[:, 1], 
              c='#8b949e', alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('Program Complexity', color='white', fontsize=12)
    ax.set_ylabel('Operator Diversity', color='white', fontsize=12)
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Add some random connections showing similar individuals
    for _ in range(30):
        idx1, idx2 = np.random.choice(len(early_data), 2, replace=False)
        if np.linalg.norm(early_data[idx1] - early_data[idx2]) < 2:
            ax.plot([early_data[idx1][0], early_data[idx2][0]], 
                   [early_data[idx1][1], early_data[idx2][1]], 
                   color='#30363d', alpha=0.3, linewidth=1)
    
    # Plot later generation (species clusters)
    ax = axes[1]
    
    # Draw species boundaries first
    from matplotlib.patches import Ellipse
    for center_x, center_y, color in species_centers:
        ellipse = Ellipse((center_x, center_y), 2.5, 2.0, 
                         facecolor=color, alpha=0.1, edgecolor=color, 
                         linewidth=2, linestyle='--')
        ax.add_patch(ellipse)
    
    # Plot individuals by species
    for point_x, point_y, color in late_data:
        ax.scatter(point_x, point_y, c=color, alpha=0.8, s=60, 
                  edgecolors='white', linewidth=0.5)
    
    # Species labels
    species_labels = [
        (-2, 3, 'Polynomial\nSpecies', '#7ee787'),
        (2, 0.2, 'Trigonometric\nSpecies', '#58a6ff'),
        (0, 2.8, 'Mixed\nSpecies', '#f97583'),
        (1, -3, 'Linear\nSpecies', '#d2a8ff'),
        (-1, -2.3, 'Exponential\nSpecies', '#ffa657'),
    ]
    
    for x, y, label, color in species_labels:
        ax.text(x, y, label, ha='center', va='center', color=color, 
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', 
                        edgecolor=color, alpha=0.8))
    
    ax.set_xlabel('Program Complexity', color='white', fontsize=12)
    ax.set_ylabel('Operator Diversity', color='white', fontsize=12)
    ax.grid(True, alpha=0.3, color='#30363d')
    
    # Add arrows showing evolution direction
    ax.annotate('', xy=(1.5, -1.5), xytext=(-0.5, 0.5),
               arrowprops=dict(arrowstyle='->', color='white', lw=2, alpha=0.7))
    ax.text(-0.5, -0.5, 'Evolution\nPressure', ha='center', va='center', 
           color='white', fontsize=9, fontweight='bold')
    
    # Statistics boxes
    early_stats = 'Generation 5:\n• Diversity: High\n• Species: 1\n• Avg Distance: 2.8'
    late_stats = 'Generation 50:\n• Diversity: Medium\n• Species: 5\n• Avg Distance: 1.2'
    
    axes[0].text(0.02, 0.98, early_stats, transform=axes[0].transAxes, 
                fontsize=10, color='white', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                         edgecolor='#30363d', alpha=0.9))
    
    axes[1].text(0.02, 0.98, late_stats, transform=axes[1].transAxes, 
                fontsize=10, color='white', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                         edgecolor='#30363d', alpha=0.9))
    
    # Legend
    legend_elements = [
        plt.scatter([], [], c='#8b949e', s=50, alpha=0.6, label='Unspecialized'),
        plt.scatter([], [], c='#7ee787', s=60, alpha=0.8, label='Polynomial'),
        plt.scatter([], [], c='#58a6ff', s=60, alpha=0.8, label='Trigonometric'),
        plt.scatter([], [], c='#f97583', s=60, alpha=0.8, label='Mixed'),
        plt.scatter([], [], c='#d2a8ff', s=60, alpha=0.8, label='Linear'),
        plt.scatter([], [], c='#ffa657', s=60, alpha=0.8, label='Exponential'),
    ]
    
    axes[1].legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.98, 0.98), facecolor='#21262d', 
                  edgecolor='#30363d', labelcolor='white', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/speciation.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_speciation()