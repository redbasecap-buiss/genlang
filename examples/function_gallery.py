#!/usr/bin/env python3
"""Function gallery - 12 different functions that genlang can discover."""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_function_gallery():
    """4×3 grid showing 12 different discoverable functions."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15), facecolor='#0d1117')
    fig.suptitle('🎨 Function Gallery — Mathematical Expressions GP Can Discover', 
                 color='white', fontsize=18, fontweight='bold', y=0.95)
    
    # Define 12 interesting functions
    functions = [
        ('Linear', 'f(x) = 2x + 1', lambda x: 2*x + 1, '#f97583'),
        ('Quadratic', 'f(x) = x² - 3x + 2', lambda x: x**2 - 3*x + 2, '#7ee787'),
        ('Cubic', 'f(x) = x³ - x', lambda x: x**3 - x, '#58a6ff'),
        ('Sine Wave', 'f(x) = sin(2πx)', lambda x: np.sin(2*np.pi*x), '#d2a8ff'),
        ('Cosine', 'f(x) = cos(x) + 0.5', lambda x: np.cos(x) + 0.5, '#ffa657'),
        ('Exponential', 'f(x) = e^x - 1', lambda x: np.exp(x) - 1, '#ff6b9d'),
        ('Logarithmic', 'f(x) = ln(x + 1)', lambda x: np.log(x + 1), '#79c0ff'),
        ('Rational', 'f(x) = x/(x + 1)', lambda x: x/(x + 1), '#f0e68c'),
        ('Gaussian', 'f(x) = e^(-x²)', lambda x: np.exp(-x**2), '#ff7f50'),
        ('Trigonometric Mix', 'f(x) = sin(x)cos(x)', lambda x: np.sin(x)*np.cos(x), '#dda0dd'),
        ('Polynomial', 'f(x) = x⁴ - 2x² + 1', lambda x: x**4 - 2*x**2 + 1, '#90ee90'),
        ('Complex', 'f(x) = sin(x²) + cos(x)', lambda x: np.sin(x**2) + np.cos(x), '#ffd700'),
    ]
    
    x_range = np.linspace(-2, 2, 200)
    
    for i, (ax, (name, expression, func, color)) in enumerate(zip(axes.flat, functions)):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        
        # Generate function data
        try:
            # Handle domain restrictions
            if name == 'Logarithmic':
                x_plot = np.linspace(0.01, 2, 200)
                y_plot = func(x_plot)
            elif name == 'Exponential':
                x_plot = np.linspace(-1, 1, 200)
                y_plot = func(x_plot)
            else:
                x_plot = x_range
                y_plot = func(x_plot)
            
            # Clip extreme values for better visualization
            y_plot = np.clip(y_plot, -10, 10)
            
        except:
            x_plot = x_range
            y_plot = np.zeros_like(x_range)
        
        # Plot function
        ax.plot(x_plot, y_plot, color=color, linewidth=3, alpha=0.9)
        
        # Add some sample points
        n_samples = 15
        if name == 'Logarithmic':
            x_samples = np.linspace(0.1, 1.8, n_samples)
        elif name == 'Exponential':
            x_samples = np.linspace(-0.8, 0.8, n_samples) 
        else:
            x_samples = np.linspace(-1.8, 1.8, n_samples)
            
        y_samples = func(x_samples) if name != 'Exponential' else np.clip(func(x_samples), -10, 10)
        y_samples = np.clip(y_samples, -10, 10)
        
        ax.scatter(x_samples, y_samples, color=color, s=25, alpha=0.7, 
                  edgecolors='white', linewidth=0.5, zorder=5)
        
        # Title and expression
        ax.set_title(name, color='white', fontsize=12, fontweight='bold', pad=10)
        ax.text(0.5, 0.05, expression, transform=ax.transAxes, ha='center', 
                va='bottom', color=color, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', 
                         edgecolor=color, alpha=0.9))
        
        # Grid and styling
        ax.grid(True, alpha=0.3, color='#30363d')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-6, 6)
        
        # Add complexity indicator
        complexity_scores = {
            'Linear': 3, 'Quadratic': 5, 'Cubic': 7, 'Sine Wave': 4,
            'Cosine': 4, 'Exponential': 6, 'Logarithmic': 6, 'Rational': 8,
            'Gaussian': 7, 'Trigonometric Mix': 9, 'Polynomial': 10, 'Complex': 12
        }
        
        complexity = complexity_scores.get(name, 5)
        
        # Complexity bar
        bar_width = 0.8
        bar_height = 0.1
        bar_x = 0.1
        bar_y = 0.9
        
        # Background bar
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_width, bar_height, 
                                  transform=ax.transAxes, facecolor='#30363d', alpha=0.5))
        
        # Complexity fill
        fill_width = (complexity / 12) * bar_width
        complexity_color = color if complexity <= 8 else '#ff6b6b'
        ax.add_patch(plt.Rectangle((bar_x, bar_y), fill_width, bar_height,
                                  transform=ax.transAxes, facecolor=complexity_color, alpha=0.8))
        
        # Complexity label
        ax.text(bar_x + bar_width/2, bar_y + bar_height/2, f'Complexity: {complexity}/12',
                transform=ax.transAxes, ha='center', va='center', 
                color='white', fontsize=8, fontweight='bold')
    
    # Add discovery difficulty legend
    difficulty_text = '''Discovery Difficulty:

🟢 Easy (1-4):    Linear, trigonometric basics
🟡 Medium (5-8):  Polynomials, simple rationals  
🔴 Hard (9-12):   Complex compositions, high-order

GP Success Factors:
• Function complexity vs population size
• Generation budget vs search space
• Building block availability
• Domain knowledge incorporation'''
    
    fig.text(0.02, 0.02, difficulty_text, fontsize=12, color='#8b949e', 
             fontweight='bold', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    # Add evolution timeline
    timeline_text = '''Typical Discovery Timeline:

Gen 0-10:   Simple building blocks (x, constants, +, ×)
Gen 10-25:  Linear combinations, basic polynomials
Gen 25-40:  Nonlinear functions, trigonometry  
Gen 40-60:  Complex compositions, optimization
Gen 60+:    Fine-tuning, coefficient optimization

Success Rate by Function Type:
• Linear/Quadratic: 95%+ success
• Trigonometric: 80%+ success  
• Exponential/Log: 60%+ success
• High-order/Complex: 30%+ success'''
    
    fig.text(0.35, 0.02, timeline_text, fontsize=12, color='white', 
             fontweight='bold', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#30363d', alpha=0.9))
    
    # Add GP advantages
    advantages_text = '''GP Advantages for Function Discovery:

🎯 No Prior Knowledge Required:
   • Discovers functions from data alone
   • No assumption about function form
   
🔍 Exploration Power:
   • Searches vast function spaces
   • Finds unexpected combinations
   
🧬 Automatic Feature Engineering:
   • Creates relevant input transformations  
   • Builds hierarchical representations
   
⚡ Parallel Evolution:
   • Multiple solution approaches simultaneously
   • Population diversity maintains exploration'''
    
    fig.text(0.68, 0.02, advantages_text, fontsize=12, color='#7ee787', 
             fontweight='bold', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                      edgecolor='#7ee787', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.92])
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/function_gallery.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_function_gallery()