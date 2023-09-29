import time
import prettytable
import torch
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training

#calculate the delta of X from source domain to achive the superium of classifier error
def pgd_linf_s_label(model, X, y, epsilon, learn_rate=0.01, num_iter=20, randomize=False):
    """ Construct PGD adversarial examples on the examples X (to achive the superium of classifier error in source domain)"""
    #when epsilon is zero, it means non-adversarial
    if epsilon == 0:
        return 0
        
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
    
    model.zero_grad()
    for t in range(num_iter):
        class_output = model(X + delta)
        err = F.cross_entropy(class_output, y)
        err.backward()
        delta.data = (delta + learn_rate*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    
    return delta.detach()

def attackstep(model, X, epsilon, num_iter=20):
    """ calculate the step number of PGD attacker (The real goal is to approximate the distance between the data and the decision boundary)"""
    origin_output = model(X)
    origin_pred = torch.max(origin_output, 1)  # we do not use the ground truth label to generate adversarial sample.
    delta = torch.zeros_like(X, requires_grad=True)
    learn_rate = epsilon / 4
    model.zero_grad()
    steps = num_iter * torch.ones_like(origin_pred[1])
    for t in range(num_iter):
        class_output = model(X + delta)
        pred = torch.max(class_output, 1)
        attack_success_index = ~(pred[1] == origin_pred[1])  # the index of data which has wrong prediction after being attacked.
        steps[attack_success_index] = torch.min(t*torch.ones_like(steps[attack_success_index]), steps[attack_success_index])
        err = F.cross_entropy(class_output, origin_pred[1])
        err.backward()
        delta.data = (delta + learn_rate*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    
    return steps