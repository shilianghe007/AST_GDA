import torch
from losses import cse
from adversarial.validate import validate, accuracy
# from torch.utils.data import TensorDataset
from data.CustomTensorDataset import CustomTensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from adversarial.attack import pgd_linf_s_label

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(optimizer, t, lr):
    """decrease the learning rate"""
    if t > 0.9:
        lr = lr * 0.005
    elif t > 0.75:
        lr = lr * 0.01
    elif t > 0.5:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_model(data, transform, model, epsilon, optimizer, epoches, batch_size, writer, t, args, present_progress=True):
    """
    input:
        data: the dataset, including four parts: training data & validation data
            the training data of the intermediate domains are features and pseudo label (set) obtained from teacher model.
        classifier_model: the convolution nn
        epsilon: the adversarial perturbation of the attacker
        optimizer: the optimizer of the model
        lr_scheduler: adjust the learning rate dynamically
        epoches: the number of epoch
        batch_size: the size of a batch in training
        writer: tensorboard
        t: the current intermediate domain index
    output:
        val_acc: the acc validate on the current domain
        val_acc_ad: the acc_ad validate on the current domain
    """
    x_train, y_train, x_val, y_val = data
    train_dataset = CustomTensorDataset(x_train, y_train, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epoches):
        if args.dy_lr:
            adjust_learning_rate(optimizer, epoch/args.epoches, args.lr)
        model.train()
        train_loss_ad = 0
        train_acc_ad = 0
        if present_progress:
            print("Epoch ", epoch)
        for _, data in enumerate(train_loader):
            x, label = data
            x = x.to(device)
            label = label.to(device)

            delta = pgd_linf_s_label(model, x, label, epsilon)
            x_ad = x + delta
            # compute output
            output = model(x)
            output_ad = model(x_ad)

            loss = cse(output, label)
            loss_ad = F.cross_entropy(output_ad, label)
            acc_ad = accuracy(output_ad, label)
            
            train_loss_ad += loss_ad.item()
            train_acc_ad += acc_ad

            optimizer.zero_grad()
            loss_ad.backward()
            optimizer.step()

        train_loss_ad = train_loss_ad / len(train_loader)
        train_acc_ad = train_acc_ad / len(train_loader)

        # val_loss, val_acc = validate(model, (x_val, y_val)) # the validation results on the current domain
        val_acc, val_acc_ad, val_loss_ad = validate(model, (x_val, y_val), args, args.epsilon)  # the validation results on the current domain
        if present_progress:
            print("train_loss_ad: %.4f, train_acc_ad: %.2f%%, val_acc: %.2f%%, val_acc_ad: %.2f%%" % (train_loss_ad, train_acc_ad * 100, val_acc * 100, val_acc_ad * 100))
        if t >= 0:
            # record the acc/loss
            writer.add_scalar('domain_{0}/train_loss'.format(t), train_loss_ad, epoch)
            writer.add_scalar('domain_{0}/train_acc'.format(t), train_acc_ad, epoch)
            writer.add_scalar('domain_{0}/val_loss'.format(t), val_loss_ad, epoch)
            writer.add_scalar('domain_{0}/val_acc'.format(t), val_acc_ad, epoch)

    ## return the loss and acc of the last epoch
    return val_acc, val_acc_ad