//! OKIR Core SSA Typed Abstract Syntax Tree.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OkirDataType {
    F32,
    F16,
    BF16,
    I32,
    I8,
    U32,
    U8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OkirOpcode {
    Add,
    Sub,
    Mul,
    Div,
    Fma,
    Rsqrt,
    Exp,
    Tanh,
    Silu,
    Gelu,
    Load,
    Store,
    WarpReduceSum,
    WarpReduceMax,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OkirInstruction {
    pub dest_reg: u32,
    pub opcode: OkirOpcode,
    pub data_type: OkirDataType,
    pub operands: Vec<u32>,
    pub immediate: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OkirFunction {
    pub name: String,
    pub instructions: Vec<OkirInstruction>,
    pub input_buffers: Vec<(String, OkirDataType)>,
    pub output_buffers: Vec<(String, OkirDataType)>,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OkirModule {
    pub name: String,
    pub functions: Vec<OkirFunction>,
}

impl OkirModule {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: Vec::new(),
        }
    }

    pub fn add_function(&mut self, func: OkirFunction) {
        self.functions.push(func);
    }
}
