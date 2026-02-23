use crate::ast::*;

/// Pretty-print an AST as an indented tree.
pub fn tree_print(node: &Node) -> String {
    let mut buf = String::new();
    tree_print_inner(node, &mut buf, "", true);
    buf
}

fn tree_print_inner(node: &Node, buf: &mut String, prefix: &str, is_last: bool) {
    let connector = if prefix.is_empty() {
        ""
    } else if is_last {
        "└── "
    } else {
        "├── "
    };

    let label = match node {
        Node::IntConst(v) => format!("{v}"),
        Node::FloatConst(v) => format!("{v:.2}"),
        Node::BoolConst(v) => format!("{v}"),
        Node::Var(i) => format!("x{i}"),
        Node::BinOp(op, _, _) => match op {
            BinOp::Add => "+".into(),
            BinOp::Sub => "-".into(),
            BinOp::Mul => "*".into(),
            BinOp::Div => "/".into(),
        },
        Node::UnaryOp(op, _) => match op {
            UnaryOp::Neg => "neg".into(),
            UnaryOp::Not => "not".into(),
        },
        Node::Cmp(op, _, _) => match op {
            CmpOp::Lt => "<".into(),
            CmpOp::Gt => ">".into(),
            CmpOp::Eq => "==".into(),
        },
        Node::If(_, _, _) => "if".into(),
        Node::MathFn(f, _) => match f {
            MathFn::Abs => "abs".into(),
            MathFn::Sqrt => "sqrt".into(),
            MathFn::Sin => "sin".into(),
            MathFn::Cos => "cos".into(),
            MathFn::Exp => "exp".into(),
            MathFn::Log => "log".into(),
        },
    };

    buf.push_str(&format!("{prefix}{connector}{label}\n"));

    let child_prefix = if prefix.is_empty() {
        String::new()
    } else if is_last {
        format!("{prefix}    ")
    } else {
        format!("{prefix}│   ")
    };

    let children: Vec<&Node> = match node {
        Node::IntConst(_) | Node::FloatConst(_) | Node::BoolConst(_) | Node::Var(_) => vec![],
        Node::BinOp(_, l, r) | Node::Cmp(_, l, r) => vec![l, r],
        Node::UnaryOp(_, c) | Node::MathFn(_, c) => vec![c],
        Node::If(a, b, c) => vec![a, b, c],
    };

    for (i, child) in children.iter().enumerate() {
        tree_print_inner(child, buf, &child_prefix, i == children.len() - 1);
    }
}

/// Export AST as a Mermaid flowchart diagram.
pub fn to_mermaid(node: &Node) -> String {
    let mut buf = String::from("graph TD\n");
    let mut counter = 0;
    to_mermaid_inner(node, &mut buf, &mut counter);
    buf
}

fn node_label(node: &Node) -> String {
    match node {
        Node::IntConst(v) => format!("{v}"),
        Node::FloatConst(v) => format!("{v:.2}"),
        Node::BoolConst(v) => format!("{v}"),
        Node::Var(i) => format!("x{i}"),
        Node::BinOp(op, _, _) => match op {
            BinOp::Add => "+".into(),
            BinOp::Sub => "-".into(),
            BinOp::Mul => "*".into(),
            BinOp::Div => "/".into(),
        },
        Node::UnaryOp(op, _) => match op {
            UnaryOp::Neg => "neg".into(),
            UnaryOp::Not => "not".into(),
        },
        Node::Cmp(op, _, _) => match op {
            CmpOp::Lt => "lt".into(),
            CmpOp::Gt => "gt".into(),
            CmpOp::Eq => "eq".into(),
        },
        Node::If(_, _, _) => "if".into(),
        Node::MathFn(f, _) => match f {
            MathFn::Abs => "abs".into(),
            MathFn::Sqrt => "sqrt".into(),
            MathFn::Sin => "sin".into(),
            MathFn::Cos => "cos".into(),
            MathFn::Exp => "exp".into(),
            MathFn::Log => "log".into(),
        },
    }
}

fn to_mermaid_inner(node: &Node, buf: &mut String, counter: &mut usize) -> usize {
    let id = *counter;
    *counter += 1;
    let label = node_label(node);
    buf.push_str(&format!("    N{id}[\"{label}\"]\n"));

    let children: Vec<&Node> = match node {
        Node::IntConst(_) | Node::FloatConst(_) | Node::BoolConst(_) | Node::Var(_) => vec![],
        Node::BinOp(_, l, r) | Node::Cmp(_, l, r) => vec![l, r],
        Node::UnaryOp(_, c) | Node::MathFn(_, c) => vec![c],
        Node::If(a, b, c) => vec![a, b, c],
    };

    for child in children {
        let child_id = to_mermaid_inner(child, buf, counter);
        buf.push_str(&format!("    N{id} --> N{child_id}\n"));
    }

    id
}

/// Render sparkline from a slice of f64 values.
pub fn sparkline(values: &[f64]) -> String {
    if values.is_empty() {
        return String::new();
    }
    let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;

    values
        .iter()
        .map(|v| {
            if range < 1e-10 {
                blocks[3]
            } else {
                let idx = (((v - min) / range) * 7.0).round() as usize;
                blocks[idx.min(7)]
            }
        })
        .collect()
}

/// Print a summary of evolution statistics.
pub fn print_evolution_summary(stats: &[crate::genetic::GenStats]) {
    if stats.is_empty() {
        return;
    }

    let fitnesses: Vec<f64> = stats.iter().map(|s| s.best_fitness).collect();
    let sizes: Vec<f64> = stats.iter().map(|s| s.avg_size).collect();

    println!("📊 Evolution Summary ({} generations)", stats.len());
    println!("   Fitness: {}", sparkline(&fitnesses));
    println!("   Bloat:   {}", sparkline(&sizes));
    println!(
        "   Best fitness: {:.6} → {:.6}",
        stats.first().unwrap().best_fitness,
        stats.last().unwrap().best_fitness
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_print() {
        let tree = Node::BinOp(
            BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::FloatConst(1.0)),
        );
        let output = tree_print(&tree);
        assert!(output.contains('+'));
        assert!(output.contains("x0"));
        assert!(output.contains("1.00"));
    }

    #[test]
    fn test_mermaid() {
        let tree = Node::BinOp(
            BinOp::Mul,
            Box::new(Node::Var(0)),
            Box::new(Node::IntConst(2)),
        );
        let mermaid = to_mermaid(&tree);
        assert!(mermaid.starts_with("graph TD"));
        assert!(mermaid.contains("N0"));
        assert!(mermaid.contains("-->"));
    }

    #[test]
    fn test_sparkline() {
        let vals = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let s = sparkline(&vals);
        assert_eq!(s.chars().count(), 8);
        assert_eq!(s.chars().next().unwrap(), '▁');
        assert_eq!(s.chars().last().unwrap(), '█');
    }

    #[test]
    fn test_sparkline_empty() {
        assert_eq!(sparkline(&[]), "");
    }

    #[test]
    fn test_sparkline_constant() {
        let s = sparkline(&[5.0, 5.0, 5.0]);
        assert_eq!(s.chars().count(), 3);
    }

    #[test]
    fn test_tree_print_nested() {
        let tree = Node::If(
            Box::new(Node::Cmp(
                CmpOp::Gt,
                Box::new(Node::Var(0)),
                Box::new(Node::IntConst(0)),
            )),
            Box::new(Node::Var(0)),
            Box::new(Node::UnaryOp(UnaryOp::Neg, Box::new(Node::Var(0)))),
        );
        let output = tree_print(&tree);
        assert!(output.contains("if"));
        assert!(output.contains('>'));
        assert!(output.contains("neg"));
    }
}
