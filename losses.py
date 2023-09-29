import torch 
import torch.nn.functional as F 

def cse(output: torch.Tensor, labels: torch.Tensor) -> float: 
    # return F.cross_entropy(output, labels, weight) 
    if labels.dim() == 1:
        return F.cross_entropy(output, labels) 
    elif labels.dim() > 1:
        return F.kl_div(F.log_softmax(output), labels)

# def l2_dis(x1: torch.Tensor, x2: torch.Tensor) -> float:
#     return torch.norm(x1-x2)