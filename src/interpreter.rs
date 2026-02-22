use crate::ast::*;

/// Runtime value type.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Value {
    pub fn to_f64(&self) -> f64 {
        match self {
            Value::Int(v) => *v as f64,
            Value::Float(v) => *v,
            Value::Bool(v) => {
                if *v {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    pub fn to_bool(&self) -> bool {
        match self {
            Value::Bool(v) => *v,
            Value::Int(v) => *v != 0,
            Value::Float(v) => *v != 0.0,
        }
    }
}

/// Execution error.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecError {
    StepLimitExceeded,
    UndefinedVariable(usize),
    InvalidValue,
}

/// Sandboxed interpreter with step limit.
pub struct Interpreter {
    pub step_limit: usize,
    steps: usize,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self {
            step_limit: 10_000,
            steps: 0,
        }
    }
}

impl Interpreter {
    pub fn new(step_limit: usize) -> Self {
        Self {
            step_limit,
            steps: 0,
        }
    }

    /// Evaluate an AST node with the given variable bindings.
    pub fn eval(&mut self, node: &Node, vars: &[f64]) -> Result<Value, ExecError> {
        self.steps += 1;
        if self.steps > self.step_limit {
            return Err(ExecError::StepLimitExceeded);
        }

        match node {
            Node::IntConst(v) => Ok(Value::Int(*v)),
            Node::FloatConst(v) => Ok(Value::Float(*v)),
            Node::BoolConst(v) => Ok(Value::Bool(*v)),
            Node::Var(i) => vars
                .get(*i)
                .map(|v| Value::Float(*v))
                .ok_or(ExecError::UndefinedVariable(*i)),
            Node::BinOp(op, l, r) => {
                let lv = self.eval(l, vars)?.to_f64();
                let rv = self.eval(r, vars)?.to_f64();
                let result = match op {
                    BinOp::Add => lv + rv,
                    BinOp::Sub => lv - rv,
                    BinOp::Mul => lv * rv,
                    BinOp::Div => {
                        if rv.abs() < 1e-10 {
                            1.0 // protected division
                        } else {
                            lv / rv
                        }
                    }
                };
                Ok(Value::Float(result))
            }
            Node::UnaryOp(op, c) => {
                let cv = self.eval(c, vars)?;
                match op {
                    UnaryOp::Neg => Ok(Value::Float(-cv.to_f64())),
                    UnaryOp::Not => Ok(Value::Bool(!cv.to_bool())),
                }
            }
            Node::Cmp(op, l, r) => {
                let lv = self.eval(l, vars)?.to_f64();
                let rv = self.eval(r, vars)?.to_f64();
                let result = match op {
                    CmpOp::Lt => lv < rv,
                    CmpOp::Gt => lv > rv,
                    CmpOp::Eq => (lv - rv).abs() < 1e-10,
                };
                Ok(Value::Bool(result))
            }
            Node::If(cond, then, els) => {
                let cv = self.eval(cond, vars)?.to_bool();
                if cv {
                    self.eval(then, vars)
                } else {
                    self.eval(els, vars)
                }
            }
            Node::MathFn(f, c) => {
                let cv = self.eval(c, vars)?.to_f64();
                let result = match f {
                    MathFn::Abs => cv.abs(),
                    MathFn::Sqrt => cv.abs().sqrt(),
                    MathFn::Sin => cv.sin(),
                    MathFn::Cos => cv.cos(),
                    MathFn::Exp => cv.min(100.0).exp(), // clamp to prevent overflow
                    MathFn::Log => cv.abs().max(1e-10).ln(),
                };
                if result.is_nan() || result.is_infinite() {
                    Ok(Value::Float(0.0))
                } else {
                    Ok(Value::Float(result))
                }
            }
        }
    }

    /// Reset step counter for reuse.
    pub fn reset(&mut self) {
        self.steps = 0;
    }

    /// Get current step count.
    pub fn steps(&self) -> usize {
        self.steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arithmetic() {
        let mut interp = Interpreter::default();
        // (x0 + 1.0) * 2.0
        let prog = Node::BinOp(
            BinOp::Mul,
            Box::new(Node::BinOp(
                BinOp::Add,
                Box::new(Node::Var(0)),
                Box::new(Node::FloatConst(1.0)),
            )),
            Box::new(Node::FloatConst(2.0)),
        );
        let result = interp.eval(&prog, &[3.0]).unwrap().to_f64();
        assert!((result - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_protected_division() {
        let mut interp = Interpreter::default();
        let prog = Node::BinOp(
            BinOp::Div,
            Box::new(Node::FloatConst(5.0)),
            Box::new(Node::FloatConst(0.0)),
        );
        let result = interp.eval(&prog, &[]).unwrap().to_f64();
        assert!((result - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_conditional() {
        let mut interp = Interpreter::default();
        // if x0 > 0 then 1 else -1
        let prog = Node::If(
            Box::new(Node::Cmp(
                CmpOp::Gt,
                Box::new(Node::Var(0)),
                Box::new(Node::FloatConst(0.0)),
            )),
            Box::new(Node::IntConst(1)),
            Box::new(Node::IntConst(-1)),
        );
        assert_eq!(interp.eval(&prog, &[5.0]).unwrap().to_f64(), 1.0);
        interp.reset();
        assert_eq!(interp.eval(&prog, &[-5.0]).unwrap().to_f64(), -1.0);
    }

    #[test]
    fn test_step_limit() {
        let mut interp = Interpreter::new(5);
        // Deep recursive tree that needs many steps
        let mut node = Node::FloatConst(1.0);
        for _ in 0..10 {
            node = Node::BinOp(BinOp::Add, Box::new(node.clone()), Box::new(node));
        }
        let result = interp.eval(&node, &[]);
        assert_eq!(result, Err(ExecError::StepLimitExceeded));
    }

    #[test]
    fn test_math_functions() {
        let mut interp = Interpreter::default();
        let prog = Node::MathFn(MathFn::Sin, Box::new(Node::FloatConst(0.0)));
        let result = interp.eval(&prog, &[]).unwrap().to_f64();
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_undefined_variable() {
        let mut interp = Interpreter::default();
        let prog = Node::Var(99);
        assert_eq!(
            interp.eval(&prog, &[]),
            Err(ExecError::UndefinedVariable(99))
        );
    }
}
