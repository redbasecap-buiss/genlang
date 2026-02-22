use rand::Rng;
use serde::{Deserialize, Serialize};

/// The core AST node representing a program genome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Node {
    /// Integer constant
    IntConst(i64),
    /// Float constant
    FloatConst(f64),
    /// Boolean constant
    BoolConst(bool),
    /// Variable reference by index (x0, x1, ...)
    Var(usize),
    /// Arithmetic: +, -, *, / (protected)
    BinOp(BinOp, Box<Node>, Box<Node>),
    /// Unary operations
    UnaryOp(UnaryOp, Box<Node>),
    /// Comparison
    Cmp(CmpOp, Box<Node>, Box<Node>),
    /// If-then-else
    If(Box<Node>, Box<Node>, Box<Node>),
    /// Built-in math function
    MathFn(MathFn, Box<Node>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div, // protected division
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CmpOp {
    Lt,
    Gt,
    Eq,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathFn {
    Abs,
    Sqrt,
    Sin,
    Cos,
    Exp,
    Log,
}

impl Node {
    /// Count total nodes in the tree.
    pub fn size(&self) -> usize {
        match self {
            Node::IntConst(_) | Node::FloatConst(_) | Node::BoolConst(_) | Node::Var(_) => 1,
            Node::BinOp(_, l, r) | Node::Cmp(_, l, r) => 1 + l.size() + r.size(),
            Node::UnaryOp(_, c) | Node::MathFn(_, c) => 1 + c.size(),
            Node::If(cond, then, els) => 1 + cond.size() + then.size() + els.size(),
        }
    }

    /// Depth of the tree.
    pub fn depth(&self) -> usize {
        match self {
            Node::IntConst(_) | Node::FloatConst(_) | Node::BoolConst(_) | Node::Var(_) => 0,
            Node::BinOp(_, l, r) | Node::Cmp(_, l, r) => 1 + l.depth().max(r.depth()),
            Node::UnaryOp(_, c) | Node::MathFn(_, c) => 1 + c.depth(),
            Node::If(a, b, c) => 1 + a.depth().max(b.depth()).max(c.depth()),
        }
    }

    /// Get a reference to the node at the given index (pre-order).
    pub fn get_node(&self, idx: usize) -> Option<&Node> {
        let mut count = 0;
        self.get_node_inner(idx, &mut count)
    }

    fn get_node_inner(&self, target: usize, count: &mut usize) -> Option<&Node> {
        if *count == target {
            return Some(self);
        }
        *count += 1;
        match self {
            Node::IntConst(_) | Node::FloatConst(_) | Node::BoolConst(_) | Node::Var(_) => None,
            Node::BinOp(_, l, r) | Node::Cmp(_, l, r) => l
                .get_node_inner(target, count)
                .or_else(|| r.get_node_inner(target, count)),
            Node::UnaryOp(_, c) | Node::MathFn(_, c) => c.get_node_inner(target, count),
            Node::If(a, b, c) => a
                .get_node_inner(target, count)
                .or_else(|| b.get_node_inner(target, count))
                .or_else(|| c.get_node_inner(target, count)),
        }
    }

    /// Replace the node at the given index (pre-order) with `replacement`.
    pub fn replace_node(&mut self, idx: usize, replacement: Node) {
        let mut count = 0;
        self.replace_node_inner(idx, replacement, &mut count);
    }

    fn replace_node_inner(&mut self, target: usize, replacement: Node, count: &mut usize) {
        if *count == target {
            *self = replacement;
            return;
        }
        *count += 1;
        match self {
            Node::IntConst(_) | Node::FloatConst(_) | Node::BoolConst(_) | Node::Var(_) => {}
            Node::BinOp(_, l, r) | Node::Cmp(_, l, r) => {
                l.replace_node_inner(target, replacement.clone(), count);
                r.replace_node_inner(target, replacement, count);
            }
            Node::UnaryOp(_, c) | Node::MathFn(_, c) => {
                c.replace_node_inner(target, replacement, count);
            }
            Node::If(a, b, c) => {
                a.replace_node_inner(target, replacement.clone(), count);
                b.replace_node_inner(target, replacement.clone(), count);
                c.replace_node_inner(target, replacement, count);
            }
        }
    }

    /// Generate a random program tree.
    pub fn random<R: Rng>(rng: &mut R, max_depth: usize, num_vars: usize) -> Node {
        if max_depth == 0 || (max_depth < 2 && rng.gen_bool(0.5)) {
            // Terminal
            Self::random_terminal(rng, num_vars)
        } else {
            // Function
            Self::random_function(rng, max_depth, num_vars)
        }
    }

    fn random_terminal<R: Rng>(rng: &mut R, num_vars: usize) -> Node {
        match rng.gen_range(0..3) {
            0 => Node::FloatConst(rng.gen_range(-10.0..10.0)),
            1 => Node::IntConst(rng.gen_range(-10..10)),
            _ => {
                if num_vars > 0 {
                    Node::Var(rng.gen_range(0..num_vars))
                } else {
                    Node::FloatConst(rng.gen_range(-10.0..10.0))
                }
            }
        }
    }

    fn random_function<R: Rng>(rng: &mut R, max_depth: usize, num_vars: usize) -> Node {
        match rng.gen_range(0..5) {
            0 => {
                let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div];
                Node::BinOp(
                    ops[rng.gen_range(0..4)],
                    Box::new(Node::random(rng, max_depth - 1, num_vars)),
                    Box::new(Node::random(rng, max_depth - 1, num_vars)),
                )
            }
            1 => {
                let ops = [CmpOp::Lt, CmpOp::Gt, CmpOp::Eq];
                Node::Cmp(
                    ops[rng.gen_range(0..3)],
                    Box::new(Node::random(rng, max_depth - 1, num_vars)),
                    Box::new(Node::random(rng, max_depth - 1, num_vars)),
                )
            }
            2 => Node::If(
                Box::new(Node::random(rng, max_depth - 1, num_vars)),
                Box::new(Node::random(rng, max_depth - 1, num_vars)),
                Box::new(Node::random(rng, max_depth - 1, num_vars)),
            ),
            3 => {
                let fns = [
                    MathFn::Abs,
                    MathFn::Sqrt,
                    MathFn::Sin,
                    MathFn::Cos,
                    MathFn::Exp,
                    MathFn::Log,
                ];
                Node::MathFn(
                    fns[rng.gen_range(0..6)],
                    Box::new(Node::random(rng, max_depth - 1, num_vars)),
                )
            }
            _ => {
                let ops = [UnaryOp::Neg, UnaryOp::Not];
                Node::UnaryOp(
                    ops[rng.gen_range(0..2)],
                    Box::new(Node::random(rng, max_depth - 1, num_vars)),
                )
            }
        }
    }

    /// Pretty print the AST as an expression string.
    pub fn to_expr(&self) -> String {
        match self {
            Node::IntConst(v) => v.to_string(),
            Node::FloatConst(v) => format!("{v:.2}"),
            Node::BoolConst(v) => v.to_string(),
            Node::Var(i) => format!("x{i}"),
            Node::BinOp(op, l, r) => {
                let sym = match op {
                    BinOp::Add => "+",
                    BinOp::Sub => "-",
                    BinOp::Mul => "*",
                    BinOp::Div => "/",
                };
                format!("({} {sym} {})", l.to_expr(), r.to_expr())
            }
            Node::UnaryOp(op, c) => {
                let sym = match op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not => "!",
                };
                format!("{sym}({})", c.to_expr())
            }
            Node::Cmp(op, l, r) => {
                let sym = match op {
                    CmpOp::Lt => "<",
                    CmpOp::Gt => ">",
                    CmpOp::Eq => "==",
                };
                format!("({} {sym} {})", l.to_expr(), r.to_expr())
            }
            Node::If(cond, then, els) => {
                format!(
                    "if {} then {} else {}",
                    cond.to_expr(),
                    then.to_expr(),
                    els.to_expr()
                )
            }
            Node::MathFn(f, c) => {
                let name = match f {
                    MathFn::Abs => "abs",
                    MathFn::Sqrt => "sqrt",
                    MathFn::Sin => "sin",
                    MathFn::Cos => "cos",
                    MathFn::Exp => "exp",
                    MathFn::Log => "log",
                };
                format!("{name}({})", c.to_expr())
            }
        }
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_expr())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_size_and_depth() {
        let tree = Node::BinOp(
            BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::FloatConst(1.0)),
        );
        assert_eq!(tree.size(), 3);
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn test_random_generation() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree = Node::random(&mut rng, 4, 2);
        assert!(tree.size() > 0);
        assert!(tree.depth() <= 4);
    }

    #[test]
    fn test_get_and_replace_node() {
        let mut tree = Node::BinOp(
            BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::IntConst(5)),
        );
        assert_eq!(tree.get_node(1), Some(&Node::Var(0)));
        tree.replace_node(1, Node::IntConst(99));
        assert_eq!(tree.get_node(1), Some(&Node::IntConst(99)));
    }

    #[test]
    fn test_display() {
        let tree = Node::BinOp(
            BinOp::Mul,
            Box::new(Node::Var(0)),
            Box::new(Node::FloatConst(3.14)),
        );
        assert_eq!(tree.to_string(), "(x0 * 3.14)");
    }

    #[test]
    fn test_serialization() {
        let tree = Node::BinOp(
            BinOp::Add,
            Box::new(Node::Var(0)),
            Box::new(Node::IntConst(1)),
        );
        let json = serde_json::to_string(&tree).unwrap();
        let back: Node = serde_json::from_str(&json).unwrap();
        assert_eq!(tree, back);
    }
}
