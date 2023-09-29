import time
import prettytable
import torch
import torch.nn.functional as F
from adversarial.attack import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training

def validate(model, data, args, epsilon, print_result=False) -> float:
    # switch to evaluate mode
    if args.per_class_eval == 1:
        confmat = ConfusionMatrix(args.num_classes)
    else:
        confmat = None
    model.eval()
    x, y = data
    dataset = TensorDataset(x, y)  # since we do not use transform at the validation period, we use TensorDataset directly.
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    acc_avg = 0
    acc_ad_avg = 0
    loss_ad_avg = 0
    # with torch.no_grad():
    for images, target in dataloader:
        images = images.to(device)
        target = target.to(device)
        
        # calculate the adversarial perturbation
        delta = pgd_linf_s_label(model, images, target, epsilon)
        images_ad = images + delta
        # compute output
        output = model(images)
        output_ad = model(images_ad)
        loss_ad = F.cross_entropy(output_ad, target)

        # measure accuracy and record loss
        acc = accuracy(output, target)
        acc_ad = accuracy(output_ad, target)
        acc_avg += acc
        acc_ad_avg += acc_ad
        loss_ad_avg += loss_ad.item()
        if confmat:
            confmat.update(target, output_ad.argmax(1))
    
    acc_avg = acc_avg / len(dataloader)
    acc_ad_avg = acc_ad_avg / len(dataloader)
    loss_ad_avg = loss_ad_avg /len(dataloader)
    if print_result:
        print('Val Acc {}, Val Robust Acc {}'.format(acc_avg, acc_ad_avg))
        if confmat:
            print(confmat.format(args.class_names))
    return acc_avg, acc_ad_avg, loss_ad_avg
    
def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for multi-class classification"""
    pred = torch.max(output, 1)
    if target.dim() > 1:
        count_acc = sum(target[torch.arange(len(pred[1])), pred[1]] > 0)  # if the target is a label set for each sample (conformal prediction)
    else:
        count_acc = sum(pred[1] == target)
    correct = count_acc / len(target)
    return correct.item()


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        """
        Update confusion matrix.

        Args:
            target: ground truth
            output: predictions of models

        Shape:
            - target: :math:`(minibatch, C)` where C means the number of classes.
            - output: :math:`(minibatch, C)` where C means the number of classes.
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        """compute global accuracy, per-class accuracy and per-class IoU"""
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

    def format(self, classes: list):
        """Get the accuracy and IoU for each class in the table format"""
        acc_global, acc, iu = self.compute()

        table = prettytable.PrettyTable(["class", "acc", "iou"])
        for i, class_name, per_acc, per_iu in zip(range(len(classes)), classes, (acc * 100).tolist(), (iu * 100).tolist()):
            table.add_row([class_name, per_acc, per_iu])

        return 'global correct: {:.1f}\nmean correct:{:.1f}\nmean IoU: {:.1f}\n{}'.format(
            acc_global.item() * 100, acc.mean().item() * 100, iu.mean().item() * 100, table.get_string())