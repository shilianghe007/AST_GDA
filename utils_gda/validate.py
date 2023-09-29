
import torch
from losses import cse
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def accuracy(output: torch.Tensor, target: torch.Tensor, return_index=False) -> float:
    """Computes the accuracy for multi-class classification"""
    pred = torch.max(output, 1)
    if target.dim() > 1:
        count_acc = sum(target[torch.arange(len(pred[1])), pred[1]] > 0)  # if the target is a label set for each sample (conformal prediction)
    else:
        index_correct = pred[1] == target
        count_acc = sum(index_correct)
    correct = count_acc / len(target)
    if return_index:
        return correct, index_correct
    else:
        return correct

def validate(model, data):
    # switch to evaluate mode
    model.eval()
    x, y = data
    with torch.no_grad():
        eva_loss = 0
        eva_acc = 0
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        for _, data in enumerate(dataloader):
            x, label = data
            x = x.to(device)
            label = label.to(device)
            predictions = model(x)
            cse_loss = cse(predictions, label)
            acc = accuracy(predictions, label)
            eva_loss += cse_loss
            eva_acc += acc

        eva_loss = eva_loss / len(dataloader)
        eva_acc = eva_acc / len(dataloader)
    return eva_loss.item(), eva_acc.item()
