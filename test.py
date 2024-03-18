import torch

torch.dot                            # [D], [D] -> []
batched_dot = torch.func.vmap(torch.dot)  # [N, D], [N, D] -> [N]
x, y = torch.ones(2, 5), torch.ones(2, 5)
print(batched_dot(x, y))