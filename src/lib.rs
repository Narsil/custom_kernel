use tch::{kind::Kind, Device, Tensor};
use torch_sys::C_tensor;

#[link(name = "fused_bloom_attention_cuda", kind = "static")]
extern "C" {
    pub fn fused_forward(
        fused_qkv: *const C_tensor,
        layer_past_key: *mut C_tensor,
        layer_past_value: *mut C_tensor,
        tensor_alibi: *const C_tensor,
        tensor_attention_mask: *const C_tensor,
        beta: f32,
        inv_norm_factor: f32,
        num_heads: i32,
        use_cache: bool,
        context_layer: *mut C_tensor,
        attention_probs: *mut C_tensor,
    );
}

