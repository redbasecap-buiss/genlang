#!/usr/bin/env python3
"""Elitism effect comparison."""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_elitism_effect():
    """Side-by-side comparison of evolution with and without elitism."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0d1117')
    fig.suptitle('⭐ Elitism Effect — Preserving Best Solutions', 
                 color='white', fontsize=16, fontweight='bold', y=0.95)
    
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
    
    generations = np.arange(0, 100)
    np.random.seed(42)
    
    # Without elitism - can lose good solutions
    no_elitism_best = []
    no_elitism_avg = []
    current_best = 50.0
    current_avg = 50.0
    
    for gen in generations:
        # Random walk - can get worse!
        best_change = np.random.normal(-0.5, 0.8)  # Generally improving but noisy
        avg_change = np.random.normal(-0.3, 0.6)
        
        # Sometimes lose good solutions (catastrophic failure)
        if np.random.random() < 0.05:  # 5% chance of losing best
            best_change = np.random.uniform(2, 8)  # Major setback
        
        current_best = max(0.1, current_best + best_change)
        current_avg = max(current_best + 2, current_avg + avg_change)
        
        no_elitism_best.append(current_best)
        no_elitism_avg.append(current_avg)
    
    # With elitism - never lose best solutions
    elitism_best = []
    elitism_avg = []
    best_ever = 50.0
    current_avg = 50.0
    
    np.random.seed(42)  # Same random seed for fair comparison
    for gen in generations:
        # Improvement attempts
        best_candidate = best_ever + np.random.normal(-0.5, 0.8)
        avg_change = np.random.normal(-0.3, 0.6)
        
        # Elitism: never get worse than current best
        best_ever = min(best_ever, max(0.05, best_candidate))
        current_avg = max(best_ever + 1, current_avg + avg_change)
        
        elitism_best.append(best_ever)
        elitism_avg.append(current_avg)
    
    # Plot without elitism
    ax = axes[0]
    ax.set_title('Without Elitism (Can Lose Good Solutions)', 
                 color='white', fontsize=12, fontweight='bold')
    
    # Fill between best and average
    ax.fill_between(generations, no_elitism_best, no_elitism_avg, 
                   color='#f97583', alpha=0.2, label='Population Range')
    
    # Plot lines
    ax.plot(generations, no_elitism_best, color='#f97583', linewidth=3, 
           label='Best Fitness', alpha=0.9)
    ax.plot(generations, no_elitism_avg, color='#f97583', linewidth=2, 
           linestyle='--', alpha=0.7, label='Average Fitness')
    
    # Highlight catastrophic failures
    failures = []
    for i in range(1, len(no_elitism_best)):
        if no_elitism_best[i] > no_elitism_best[i-1] + 1:  # Significant worsening
            failures.append(i)
    
    for failure in failures[:3]:  # Highlight first 3 failures
        ax.scatter(failure, no_elitism_best[failure], color='#ff6b6b', 
                  s=100, marker='X', zorder=5, alpha=0.9)
        ax.annotate('Lost best!', xy=(failure, no_elitism_best[failure]), 
                   xytext=(failure+5, no_elitism_best[failure]+5),
                   color='#ff6b6b', fontsize=9, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#ff6b6b', alpha=0.8))
    
    ax.set_xlabel('Generation', color='white', fontsize=12)
    ax.set_ylabel('Fitness (MSE)', color='white', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, color='#30363d')
    ax.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d', 
              labelcolor='white', fontsize=10)
    
    # Plot with elitism
    ax = axes[1]
    ax.set_title('With Elitism (Best Solutions Preserved)', 
                 color='white', fontsize=12, fontweight='bold')
    
    # Fill between best and average
    ax.fill_between(generations, elitism_best, elitism_avg, 
                   color='#7ee787', alpha=0.2, label='Population Range')
    
    # Plot lines
    ax.plot(generations, elitism_best, color='#7ee787', linewidth=3, 
           label='Best Fitness (Protected)', alpha=0.9)
    ax.plot(generations, elitism_avg, color='#7ee787', linewidth=2, 
           linestyle='--', alpha=0.7, label='Average Fitness')
    
    # Highlight steady improvements
    improvements = []
    for i in range(1, len(elitism_best)):
        if elitism_best[i-1] - elitism_best[i] > 2:  # Significant improvement
            improvements.append(i)
    
    for improvement in improvements[:3]:  # Highlight first 3 major improvements
        ax.scatter(improvement, elitism_best[improvement], color='#58a6ff', 
                  s=100, marker='*', zorder=5, alpha=0.9)
        ax.annotate('Breakthrough!', xy=(improvement, elitism_best[improvement]), 
                   xytext=(improvement+5, elitism_best[improvement]*0.7),
                   color='#58a6ff', fontsize=9, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='#58a6ff', alpha=0.8))
    
    ax.set_xlabel('Generation', color='white', fontsize=12)
    ax.set_ylabel('Fitness (MSE)', color='white', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, color='#30363d')
    ax.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d', 
              labelcolor='white', fontsize=10)
    
    # Performance comparison
    final_no_elitism = no_elitism_best[-1]
    final_elitism = elitism_best[-1]
    
    # Count setbacks
    setbacks_no_elitism = sum(1 for i in range(1, len(no_elitism_best)) 
                             if no_elitism_best[i] > no_elitism_best[i-1])
    setbacks_elitism = sum(1 for i in range(1, len(elitism_best)) 
                          if elitism_best[i] > elitism_best[i-1])
    
    comparison_text = f'''Performance Comparison:

Metric                Without Elitism    With Elitism
───────────────────────────────────────────────────
Final Best Fitness    {final_no_elitism:.3f}              {final_elitism:.3f}
Performance Setbacks  {setbacks_no_elitism}                  {setbacks_elitism}
Stability             Unstable           Monotonic
Convergence Rate      Variable           Consistent

Improvement: {((final_no_elitism - final_elitism)/final_no_elitism)*100:.0f}% better with elitism'''
    
    fig.text(0.02, 0.02, comparison_text, fontsize=11, color='white', 
             fontweight='bold', va='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    # Elitism strategy explanation
    strategy_text = '''Elitism Strategies:

🔴 No Elitism:
• All individuals subject to selection
• Can randomly lose best solutions
• Higher exploration, unstable progress

🟢 With Elitism:
• Top N% automatically survive
• Monotonic improvement guaranteed
• Stable convergence, less exploration

⚖️ Trade-offs:
• Elitism: Faster convergence, local optima risk
• No elitism: Better exploration, unstable progress
• Hybrid: Partial elitism (e.g., top 5%) balances both'''
    
    fig.text(0.55, 0.02, strategy_text, fontsize=11, color='#8b949e', 
             fontweight='bold', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.35, 1, 0.92])
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/elitism_effect.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_elitism_effect()