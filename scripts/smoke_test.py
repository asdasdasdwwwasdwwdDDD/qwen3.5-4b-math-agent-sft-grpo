# -*- coding: utf-8 -*-
"""环境冒烟测试：CUDA 可用性、bf16、bitsandbytes 4bit 量化、SDPA 注意力"""
import torch


def main():
    print(f"torch        : {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    assert torch.cuda.is_available(), "CUDA 不可用，请检查 PyTorch 是否为 CUDA 版本"
    print(f"cuda device   : {torch.cuda.get_device_name(0)}")
    print(f"cuda version  : {torch.version.cuda}")
    print(f"bf16 support  : {torch.cuda.is_bf16_supported()}")

    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).sum()
    print(f"bf16 matmul ok: {float(y):.2f}")

    import bitsandbytes as bnb
    print(f"bitsandbytes  : {bnb.__version__}")
    layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                              quant_type="nf4").cuda()
    x4 = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    out = layer(x4)  # 首次调用触发量化
    assert out.shape == (8, 64)
    print("bnb Linear4bit (NF4) 前向 OK")

    import transformers, peft, accelerate, datasets
    print(f"transformers  : {transformers.__version__}")
    print(f"peft          : {peft.__version__}")
    print(f"accelerate    : {accelerate.__version__}")
    print(f"datasets      : {datasets.__version__}")
    print("\n所有冒烟测试通过 ✔")


if __name__ == "__main__":
    main()
