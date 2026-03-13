import torch
import time

# Create two large matrices to simulate Transformer-style math
sz = 4096
a = torch.randn(sz, sz, device='cuda')
b = torch.randn(sz, sz, device='cuda')

def benchmark(precision_level):
    torch.set_float32_matmul_precision(precision_level)
    # Warmup
    for _ in range(10): a @ b
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100):
        c = a @ b
    torch.cuda.synchronize()
    return (time.time() - start) / 100

t_highest = benchmark('highest') # Standard FP32
t_high = benchmark('high')       # TF32 enabled

print(f"FP32 Time: {t_highest:.4f}s")
print(f"TF32 ('high') Time: {t_high:.4f}s")
print(f"Speedup: {t_highest / t_high:.2f}x")