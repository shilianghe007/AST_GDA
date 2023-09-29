from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensor_X, tensor_Y, transform=None):
        assert (tensor_X.size(0) == tensor_Y.size(0))
        self.tensor_X = tensor_X
        self.tensor_Y = tensor_Y
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensor_X[index]

        if self.transform:
            x = self.transform(x)

        y = self.tensor_Y[index]

        return x, y

    def __len__(self):
        return self.tensor_X.size(0)
