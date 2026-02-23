#!/usr/bin/env python3
"""Building blocks hypothesis visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import os

def create_building_blocks():
    """Show building blocks hypothesis - useful subtrees preserved and combined."""
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    ax.set_title('🧩 Building Blocks Hypothesis — Useful Subtrees Preserved & Combined', 
                 color='white', fontsize=18, fontweight='bold', pad=30)
    
    # Define useful building blocks (common patterns)
    building_blocks = [
        # Block 1: x² pattern
        {
            'name': 'Square Block',
            'expression': 'x²',
            'nodes': [(1, 2, '×', '#58a6ff'), (0.5, 1, 'x', '#7ee787'), (1.5, 1, 'x', '#7ee787')],
            'edges': [(0, 1), (0, 2)],
            'position': (2, 8),
            'frequency': 85,
            'color': '#58a6ff'
        },
        # Block 2: Linear term
        {
            'name': 'Linear Block', 
            'expression': 'a×x',
            'nodes': [(1, 2, '×', '#f97583'), (0.5, 1, 'a', '#d2a8ff'), (1.5, 1, 'x', '#7ee787')],
            'edges': [(0, 1), (0, 2)],
            'position': (6, 8),
            'frequency': 92,
            'color': '#f97583'
        },
        # Block 3: Trigonometric
        {
            'name': 'Trig Block',
            'expression': 'sin(x)',
            'nodes': [(1, 2, 'sin', '#d2a8ff'), (1, 1, 'x', '#7ee787')],
            'edges': [(0, 1)],
            'position': (10, 8),
            'frequency': 67,
            'color': '#d2a8ff'
        },
        # Block 4: Constant
        {
            'name': 'Constant Block',
            'expression': 'c',
            'nodes': [(1, 1.5, 'c', '#ffa657')],
            'edges': [],
            'position': (14, 8),
            'frequency': 78,
            'color': '#ffa657'
        }
    ]
    
    # Draw building blocks
    for block in building_blocks:
        x_pos, y_pos = block['position']
        
        # Background box
        bbox = FancyBboxPatch((x_pos - 1.2, y_pos - 1.8), 2.4, 2.5,
                             boxstyle='round,pad=0.1', 
                             facecolor=block['color'], alpha=0.1,
                             edgecolor=block['color'], linewidth=2)
        ax.add_patch(bbox)
        
        # Block name and expression
        ax.text(x_pos, y_pos + 1, block['name'], ha='center', va='center',
                color=block['color'], fontsize=12, fontweight='bold')
        ax.text(x_pos, y_pos - 1.5, block['expression'], ha='center', va='center',
                color='white', fontsize=11, fontweight='bold')
        
        # Frequency indicator
        ax.text(x_pos + 1, y_pos + 0.8, f"{block['frequency']}%", 
                ha='center', va='center', color=block['color'], 
                fontsize=10, fontweight='bold')
        
        # Draw mini AST
        nodes = [(x_pos + nx - 1, y_pos + ny - 2, label, color) 
                for nx, ny, label, color in block['nodes']]
        edges = block['edges']
        
        # Draw edges
        for parent_idx, child_idx in edges:
            if parent_idx < len(nodes) and child_idx < len(nodes):
                px, py = nodes[parent_idx][0], nodes[parent_idx][1]
                cx, cy = nodes[child_idx][0], nodes[child_idx][1]
                ax.plot([px, cx], [py, cy], color='#30363d', linewidth=1.5)
        
        # Draw nodes
        for nx, ny, label, color in nodes:
            circle = Circle((nx, ny), 0.15, facecolor=color, edgecolor='white', 
                           linewidth=1, alpha=0.9)
            ax.add_patch(circle)
            ax.text(nx, ny, label, ha='center', va='center', fontsize=9, 
                   fontweight='bold', color='white')
    
    # Show combination examples
    combinations = [
        {
            'blocks': ['Square Block', 'Constant Block'],
            'result': 'x² + c',
            'position': (4, 5),
            'fitness': 0.12
        },
        {
            'blocks': ['Square Block', 'Linear Block', 'Constant Block'],
            'result': 'x² + a×x + c',
            'position': (8, 5),
            'fitness': 0.03
        },
        {
            'blocks': ['Trig Block', 'Linear Block'],
            'result': 'sin(x) + a×x',
            'position': (12, 5),
            'fitness': 0.08
        }
    ]
    
    # Draw combination arrows and results
    for i, combo in enumerate(combinations):
        x_pos, y_pos = combo['position']
        
        # Result expression
        result_box = FancyBboxPatch((x_pos - 1.5, y_pos - 0.5), 3, 1,
                                   boxstyle='round,pad=0.1', 
                                   facecolor='#7ee787', alpha=0.2,
                                   edgecolor='#7ee787', linewidth=2)
        ax.add_patch(result_box)
        
        ax.text(x_pos, y_pos, combo['result'], ha='center', va='center',
                color='white', fontsize=12, fontweight='bold')
        ax.text(x_pos, y_pos - 0.7, f'Fitness: {combo["fitness"]}', 
                ha='center', va='center', color='#7ee787', 
                fontsize=10, fontweight='bold')
        
        # Draw arrows from building blocks to combination
        for j, block_name in enumerate(combo['blocks']):
            # Find source block
            source_block = next(b for b in building_blocks if b['name'] == block_name)
            sx, sy = source_block['position']
            
            # Curved arrow
            ax.annotate('', xy=(x_pos + j*0.3 - 0.3, y_pos + 0.5), 
                       xytext=(sx, sy - 1.8),
                       arrowprops=dict(arrowstyle='->', color=source_block['color'], 
                                     alpha=0.7, connectionstyle='arc3,rad=0.3'))
    
    # Evolution timeline showing building block discovery
    timeline_y = 2
    generations = [5, 15, 25, 35, 45]
    events = [
        'Random initialization',
        'First building blocks emerge',
        'Block preservation begins', 
        'Complex combinations appear',
        'Optimal solution assembled'
    ]
    
    ax.text(8, 3, 'Building Block Evolution Timeline', ha='center', va='center',
            color='white', fontsize=14, fontweight='bold')
    
    for i, (gen, event) in enumerate(zip(generations, events)):
        x = 2 + i * 3
        
        # Timeline point
        ax.scatter(x, timeline_y, s=100, color='#58a6ff', edgecolor='white', 
                  linewidth=2, zorder=5)
        ax.text(x, timeline_y, f'{gen}', ha='center', va='center', 
               color='white', fontsize=9, fontweight='bold')
        
        # Event description
        ax.text(x, timeline_y - 0.8, event, ha='center', va='center',
                color='#8b949e', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#21262d', 
                         edgecolor='#30363d', alpha=0.8))
        
        # Connect timeline points
        if i < len(generations) - 1:
            ax.plot([x + 0.2, x + 2.8], [timeline_y, timeline_y], 
                   color='#58a6ff', linewidth=2, alpha=0.6)
    
    # Building blocks theory explanation
    theory_text = '''Building Blocks Theory:

1. 🧩 Identification: GP discovers small, useful subtrees (building blocks)
2. 🔒 Preservation: Selection pressure maintains these blocks in population  
3. 🔗 Combination: Crossover combines blocks into larger, more complex solutions
4. 🎯 Optimization: Successful combinations become new building blocks
5. 🏆 Assembly: Final solution is optimal combination of proven blocks

Key Principles:
• Low-order building blocks form first (simple patterns)
• Selection preserves above-average building blocks
• Crossover respects block boundaries when possible
• Mutation can create new building blocks'''
    
    ax.text(0.5, 1, theory_text, fontsize=11, color='#8b949e', 
           fontweight='bold', va='bottom',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                    edgecolor='#30363d', alpha=0.9))
    
    # Success metrics
    success_text = '''Building Block Success Metrics:

🔍 Discovery Rate: 
   • Simple blocks (1-3 nodes): 85-95%
   • Complex blocks (4+ nodes): 45-70%

⏱️ Preservation Time:
   • Useful blocks: 20+ generations  
   • Random patterns: 2-5 generations

🎯 Combination Success:
   • 2 blocks → 78% improvement chance
   • 3+ blocks → 45% improvement chance

📈 Evolution Efficiency:
   • With blocks: 60% faster convergence
   • Without blocks: Random search behavior'''
    
    ax.text(9.5, 1, success_text, fontsize=11, color='white', 
           fontweight='bold', va='bottom',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', 
                    edgecolor='#30363d', alpha=0.9))
    
    plt.tight_layout()
    out = os.path.expanduser('~/.openclaw/workspace/genlang/examples/building_blocks.png')
    plt.savefig(out, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")
    return out

if __name__ == '__main__':
    create_building_blocks()