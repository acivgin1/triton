import torch
import triton
import triton.language as tl

@triton.jit
def exp(x):
    tl.store(x, tl.exp(tl.load(x)))

def exp_launcher(x):
    N = x.numel()
    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK']),)

    exp[N](x)
    return x

if __name__ == "__main__":
    x = torch.rand(100, 100, device="cuda")
    print("created x")
    exp_launcher(x)
