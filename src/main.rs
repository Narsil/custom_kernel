use tch::{kind::Kind, Device, Tensor};
use custom_kernel::fused_forward;

fn main() {
    let B = 1;
    let Q = 1;

    let S = 4;
    let H = 1;
    let NH = 1;
    let HD = H / NH;

    let fused_qkv = Tensor::ones(&[B, Q, 3 * H], (Kind::Float, Device::Cuda(0)));
    let mut past_key = Tensor::ones(&[B * NH, HD, S], (Kind::Float, Device::Cuda(0)));
    let mut past_value = Tensor::ones(&[B * NH, S, HD], (Kind::Float, Device::Cuda(0)));
    let alibi = Tensor::ones(&[B * NH, Q, S + Q], (Kind::Float, Device::Cuda(0)));
    let attention_mask = Tensor::ones(&[B, B, S + Q], (Kind::Bool, Device::Cuda(0)));
    let mut context_layer = Tensor::ones(&[B, S + Q, H], (Kind::Float, Device::Cuda(0)));
    let mut attention_probs = Tensor::ones(&[B * NH, S + Q], (Kind::Float, Device::Cuda(0)));
    println!("Fused qkv {fused_qkv:?}");
    println!("Past key {past_key:?}");
    println!("Past value {past_value:?}");
    println!("alibi {alibi:?}");
    println!("attention mask {attention_mask:?}");
    println!("context {context_layer:?}");
    println!("attention_probs {attention_probs:?}");
    unsafe {
        fused_forward(
            fused_qkv.as_ptr(),
            past_key.as_mut_ptr(),
            past_value.as_mut_ptr(),
            alibi.as_ptr(),
            attention_mask.as_ptr(),
            1.0,
            0.125,
            NH as i32,
            true,
            context_layer.as_mut_ptr(),
            attention_probs.as_mut_ptr(),
        );
    }
    println!("context {context_layer:?}");
    println!("attention_probs {attention_probs:?}");
    println!("past_key {past_key:?}");
    println!("past_value {past_value:?}");
    println!("context {:?}", Vec::<f64>::from(context_layer));
    println!("attention probs {:?}", Vec::<f64>::from(attention_probs));
}
