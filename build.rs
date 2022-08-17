use std::env;

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    println!("cargo:rerun-if-env-changed={}", name);
    env::var(name)
}

fn main() {
    let cuda_home = env_var_rerun("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda/".to_string());
    let libtorch = env_var_rerun("LIBTORCH").unwrap();
    cc::Build::new()
        // Switch to CUDA C++ library compilation using NVCC.
        .cuda(true)
        .cudart("static")
        .pic(true)
        .flag("-std=c++17")
        .flag("--expt-relaxed-constexpr")
        // Generate code for Pascal (Jetson TX2).
        .include(&format!("{libtorch}/include"))
        .include(
            &format!("{libtorch}/include/torch/csrc/api/include/"),
        )
        .flag("-gencode")
        .flag("arch=compute_80,code=sm_80")
        .file("fused_bloom_attention_cuda.cu")
        .warnings(false)
        .compile("libfused_bloom_attention_cuda.a");
    println!("cargo:rerun-if-changed=fused_bloom_attention_cuda.cu");
    println!("cargo:rustc-link-search=native={cuda_home}lib64/");
    println!("cargo:rustc-link-search=native={libtorch}lib/");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=torch_cuda_cu");
    println!("cargo:rustc-link-lib=torch_cuda_cpp");
    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=c10");
}
