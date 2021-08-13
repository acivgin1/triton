import torch
import triton
import triton.language as tl


@triton.jit
def exp(x):
    idx = tl.program_id(0)
    tl.store(x+idx, tl.exp(tl.load(x+idx)))


def exp_launcher(x):
    N = x.numel()

    exp[(N,)](x)
    return x


if __name__ == "__main__":
    x = torch.rand(100, 100, device="cuda", dtype=torch.float32)
    triton_exp_x = exp_launcher(x.clone())
    torch_exp_x = torch.exp(x)
    distance = torch.dist(triton_exp_x, torch_exp_x)
    print(f"Distance between torch.exp and triton.exp is {distance}")
