import torch

## randomly flip some labels into the other labels
def noisify_label(labels, noisy_type, noisy_rate, K):
    flip_index = torch.rand(labels.shape) < noisy_rate
    flip_len = sum(flip_index)
    if noisy_type == 'symmetric':
        rand_bias = torch.randint(1, K, (flip_len,))
    elif noisy_type == 'asymmetric':    
        rand_bias = torch.ones((flip_len,))
    labels[flip_index] += rand_bias
    labels = labels % K
    return labels