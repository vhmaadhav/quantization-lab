from safetensors import safe_open

model_path = r"models\hf\Qwen3-1.7B-Base\model.safetensors"
tensor_name = "model.layers.0.self_attn.q_proj.weight"

with safe_open(model_path, framework="pt", device="cpu") as f:
    w = f.get_tensor(tensor_name)

print("Tensor:", tensor_name)
print("Shape:", w.shape)
print("Stored dtype:", w.dtype)

print("\nFirst 16 BF16 weights:")
print(w[0, :16])

print("\nSame values converted to FP32 for easier calculations:")
print(w[0, :16].float())