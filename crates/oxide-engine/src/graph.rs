use oxide_core::error::Result;
use oxide_models::loader::ModelMetadata;

/// Operation codes for dynamic compute graph nodes (C-style enum, zero vtable overhead).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpCode {
    MulMat,         // Matrix Multiplication (GEMV / GEMM)
    Rope,           // Rotary Position Embedding
    RmsNorm,        // RMS Normalization
    FlashAttn,      // Paged Flash Attention
    Softmax,        // Softmax
    SwiGlu,         // SwiGLU FFN activation
    Add,            // Residual addition
    FusedRmsMulMat, // Secret weapon: Fused RMSNorm + GEMV
    FusedSwiGluMul, // Fused SwiGLU + Down Projection
}

pub type TensorId = u32;

/// Parameters associated with a GraphNode.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeParams {
    pub eps: f32,
    pub theta: f32,
    pub position: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Default for NodeParams {
    fn default() -> Self {
        Self {
            eps: 1e-5,
            theta: 500_000.0,
            position: 0,
            in_dim: 0,
            out_dim: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
        }
    }
}

/// A single node in the execution DAG.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub op: OpCode,
    pub src0: TensorId,
    pub src1: TensorId,
    pub dst: TensorId,
    pub dst_size: usize,
    pub weight_name: Option<String>,
    pub params: NodeParams,
}

/// Dynamic Compute Graph representing the execution plan.
#[derive(Debug, Clone, Default)]
pub struct ComputeGraph {
    pub nodes: Vec<GraphNode>,
    pub tensor_count: usize,
}

impl ComputeGraph {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(
        &mut self,
        op: OpCode,
        src0: TensorId,
        src1: TensorId,
        dst_size: usize,
        weight_name: Option<String>,
        params: NodeParams,
    ) -> TensorId {
        let dst = self.tensor_count as TensorId;
        self.tensor_count += 1;
        self.nodes.push(GraphNode {
            op,
            src0,
            src1,
            dst,
            dst_size,
            weight_name,
            params,
        });
        dst
    }

    /// Runtime Graph Fusion pass: looks for sequential patterns and fuses operations
    /// (e.g. RmsNorm + MulMat -> FusedRmsMulMat), cutting memory roundtrips by 30-50%.
    pub fn optimize_fusions(&mut self) {
        let mut fused_nodes = Vec::with_capacity(self.nodes.len());
        let mut i = 0;
        while i < self.nodes.len() {
            if i + 1 < self.nodes.len() {
                let n1 = &self.nodes[i];
                let n2 = &self.nodes[i + 1];
                if n1.op == OpCode::RmsNorm
                    && n2.op == OpCode::MulMat
                    && n1.dst == n2.src0
                {
                    let mut params = n2.params.clone();
                    params.eps = n1.params.eps;
                    fused_nodes.push(GraphNode {
                        op: OpCode::FusedRmsMulMat,
                        src0: n1.src0,
                        src1: n2.src1,
                        dst: n2.dst,
                        dst_size: n2.dst_size,
                        weight_name: n2.weight_name.clone(),
                        params,
                    });
                    i += 2;
                    continue;
                }
            }
            fused_nodes.push(self.nodes[i].clone());
            i += 1;
        }
        self.nodes = fused_nodes;
    }
}

/// Builds execution compute graph dynamically from model architecture metadata.
#[must_use]
pub fn build_graph(arch: &str, meta: &ModelMetadata, batch_size: usize) -> ComputeGraph {
    let lower_arch = arch.to_lowercase();
    match lower_arch.as_str() {
        "llama" | "qwen2" | "mistral" | "deepseek" | "gemma" => {
            let mut graph = build_transformer_graph(meta, batch_size);
            graph.optimize_fusions();
            graph
        }
        _ => {
            let mut graph = build_transformer_graph(meta, batch_size);
            graph.optimize_fusions();
            graph
        }
    }
}

/// Constructs Dense Transformer execution DAG.
#[must_use]
pub fn build_transformer_graph(meta: &ModelMetadata, _batch_size: usize) -> ComputeGraph {
    let mut graph = ComputeGraph::new();
    let h = meta.hidden_size as usize;
    let num_heads = meta.num_heads as usize;
    let num_kv_heads = meta.num_kv_heads as usize;
    let head_dim = meta.head_dim as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let inter_dim = meta.intermediate_size as usize;

    // Initial input activation token (src 0)
    let mut current_hidden = 0u32;
    graph.tensor_count = 1;

    for layer in 0..meta.num_layers {
        // 1. Attention RMSNorm
        let attn_norm_params = NodeParams {
            eps: meta.rms_norm_eps,
            ..Default::default()
        };
        let norm_attn = graph.add_node(
            OpCode::RmsNorm,
            current_hidden,
            0,
            h,
            Some(format!("blk.{layer}.attn_norm.weight")),
            attn_norm_params,
        );

        // 2. Q, K, V Projections
        let q_params = NodeParams {
            in_dim: h,
            out_dim: q_dim,
            ..Default::default()
        };
        let q = graph.add_node(
            OpCode::MulMat,
            norm_attn,
            0,
            q_dim,
            Some(format!("blk.{layer}.attn_q.weight")),
            q_params,
        );

        let k_params = NodeParams {
            in_dim: h,
            out_dim: kv_dim,
            ..Default::default()
        };
        let k = graph.add_node(
            OpCode::MulMat,
            norm_attn,
            0,
            kv_dim,
            Some(format!("blk.{layer}.attn_k.weight")),
            k_params,
        );

        let v_params = NodeParams {
            in_dim: h,
            out_dim: kv_dim,
            ..Default::default()
        };
        let v = graph.add_node(
            OpCode::MulMat,
            norm_attn,
            0,
            kv_dim,
            Some(format!("blk.{layer}.attn_v.weight")),
            v_params,
        );

        // 3. Rotary Position Embeddings (RoPE)
        let rope_params = NodeParams {
            theta: meta.rope_freq_base,
            num_heads,
            num_kv_heads,
            head_dim,
            ..Default::default()
        };
        let q_rope = graph.add_node(OpCode::Rope, q, 0, q_dim, None, rope_params.clone());
        let k_rope = graph.add_node(OpCode::Rope, k, 0, kv_dim, None, rope_params);

        // 4. Flash Attention (GQA)
        let attn_params = NodeParams {
            num_heads,
            num_kv_heads,
            head_dim,
            ..Default::default()
        };
        let attn_out = graph.add_node(OpCode::FlashAttn, q_rope, k_rope, q_dim, None, attn_params);

        // 5. Output Projection (o_proj)
        let o_params = NodeParams {
            in_dim: q_dim,
            out_dim: h,
            ..Default::default()
        };
        let o_proj = graph.add_node(
            OpCode::MulMat,
            attn_out,
            0,
            h,
            Some(format!("blk.{layer}.attn_output.weight")),
            o_params,
        );

        // 6. Residual 1 (hidden + o_proj)
        let res_1 = graph.add_node(OpCode::Add, current_hidden, o_proj, h, None, NodeParams::default());

        // 7. FFN RMSNorm
        let ffn_norm_params = NodeParams {
            eps: meta.rms_norm_eps,
            ..Default::default()
        };
        let norm_ffn = graph.add_node(
            OpCode::RmsNorm,
            res_1,
            0,
            h,
            Some(format!("blk.{layer}.ffn_norm.weight")),
            ffn_norm_params,
        );

        // 8. Gate and Up Projections
        let gate_params = NodeParams {
            in_dim: h,
            out_dim: inter_dim,
            ..Default::default()
        };
        let gate = graph.add_node(
            OpCode::MulMat,
            norm_ffn,
            0,
            inter_dim,
            Some(format!("blk.{layer}.ffn_gate.weight")),
            gate_params,
        );

        let up_params = NodeParams {
            in_dim: h,
            out_dim: inter_dim,
            ..Default::default()
        };
        let up = graph.add_node(
            OpCode::MulMat,
            norm_ffn,
            0,
            inter_dim,
            Some(format!("blk.{layer}.ffn_up.weight")),
            up_params,
        );

        // 9. SwiGLU activation
        let swiglu = graph.add_node(OpCode::SwiGlu, gate, up, inter_dim, None, NodeParams::default());

        // 10. Down Projection
        let down_params = NodeParams {
            in_dim: inter_dim,
            out_dim: h,
            ..Default::default()
        };
        let down = graph.add_node(
            OpCode::MulMat,
            swiglu,
            0,
            h,
            Some(format!("blk.{layer}.ffn_down.weight")),
            down_params,
        );

        // 11. Residual 2 (res_1 + down)
        current_hidden = graph.add_node(OpCode::Add, res_1, down, h, None, NodeParams::default());
    }

    // Final RMSNorm
    let final_norm_params = NodeParams {
        eps: meta.rms_norm_eps,
        ..Default::default()
    };
    let final_norm = graph.add_node(
        OpCode::RmsNorm,
        current_hidden,
        0,
        h,
        Some("output_norm.weight".to_string()),
        final_norm_params,
    );

    // LM Head
    let vocab_size = meta.vocab_size as usize;
    let lm_head_params = NodeParams {
        in_dim: h,
        out_dim: vocab_size,
        ..Default::default()
    };
    graph.add_node(
        OpCode::MulMat,
        final_norm,
        0,
        vocab_size,
        Some("output.weight".to_string()),
        lm_head_params,
    );

    graph
}
