from safetensors import safe_open

model_path = r"models\hf\Qwen3-1.7B-Base\model.safetensors"

with safe_open(model_path, framework="pt", device="cpu") as f:
    keys = list(f.keys())

    print("Number of tensors:", len(keys))
    print()

    for name in keys[:20]:
        tensor = f.get_tensor(name)

        print(
            f"{name:70} "
            f"shape={str(tuple(tensor.shape)):25} "
            f"dtype={tensor.dtype}"
        )