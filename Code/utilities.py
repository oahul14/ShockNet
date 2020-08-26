import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed):
    """Set random seed for all possible random generators

    Args:
        seed (int): Seed to be set.

    Returns:
        Bool: Status if seed has been set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.enabled   = False
    return True

def set_device():
    """Set device. GPU of higher priority.

    Returns:
        int: device name.
    """
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        print("GPU name: ", torch.cuda.get_device_name())
        device = 'cuda'
    else:
        print("No GPU available!")
    return device

def get_mean_std(dataloader):
    """Compute mean and std on the fly.


    Args:
        dataloader (Dataloader): Dataloader class from torch.utils.data.

    Returns:
        ndarray: ndarray of mean and std.
    """
    cnt = 0
    mean = 0
    std = 0
    for l in dataloader:                        # Now in (batch, channel, h, w)
        data = l[0].double()                    # set dtype
        b = data.size(0)                        # batch size at axis=0
        data = data.view(b, data.size(1), -1)   # reshape the tensor into (b, channel, h, w)
        mean += data.mean(2).sum(0)             # calculate mean for 3 channels
        std += data.std(2).sum(0)               # calculate std for 3 channels
        cnt += b                                # get the count of data
    mean /= cnt
    std /= cnt
    return mean.cpu().detach().numpy(), std.cpu().detach().numpy()

def kaiming_uniform_init(m):
    """Initialise the model with Kaiming uniform distribution.

    Args:
        m (torch model): Neural network class in torch.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

def kaiming_normal_init(m):
    """Initialise the model with Kaiming normal distribution.

    Args:
        m (torch model): Neural network class in torch.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")




