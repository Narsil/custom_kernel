use torch_sys::C_tensor;

extern "C" {
    fn fused_forward(
        fused_qkv: *const C_tensor,
        layer_past_key: *mut C_tensor,
        layer_past_value: *mut C_tensor,
        tensor_alibi: *const C_tensor,
        tensor_attention_mask: *const C_tensor,
        head_mask: *const C_tensor,
        beta: f32,
        inv_norm_factor: f32,
        num_heads: i32,
        use_cache: bool,
        context_layer: *mut C_tensor,
        attention_probs: *mut C_tensor,
    );
}


#[cfg(test)]
mod tests{
    use super::*;
    use tch::{Tensor, kind::Kind, Device};

    #[test]
    fn test(){
        let fused_qkv = Tensor::ones(&[1, 1], (Kind::Float, Device::Cuda(0)));
        let mut past_key = Tensor::ones(&[1, 1], (Kind::Float, Device::Cuda(0)));
        let mut past_value = Tensor::ones(&[1, 1], (Kind::Float, Device::Cuda(0)));
        let alibi = Tensor::ones(&[1, 1], (Kind::Float, Device::Cuda(0)));
        let attention_mask = Tensor::ones(&[1, 1], (Kind::Float, Device::Cuda(0)));
        let head_mask = Tensor::ones(&[1, 1], (Kind::Float, Device::Cuda(0)));
        let mut context_layer = Tensor::ones(&[1, 1], (Kind::Float, Device::Cuda(0)));
        let mut attention_probs = Tensor::ones(&[1, 1], (Kind::Float, Device::Cuda(0)));
    unsafe{
        fused_forward(fused_qkv.as_ptr(), past_key.as_mut_ptr(), past_value.as_mut_ptr(), alibi.as_ptr(), attention_mask.as_ptr(), head_mask.as_ptr(), 1.0, 0.125, 16, true, context_layer.as_mut_ptr(), attention_probs.as_mut_ptr(), );
    }
    }
}
